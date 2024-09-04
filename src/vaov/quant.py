import jax
import jaxlib
from dataclasses import dataclass
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
import jax
from math import ceil
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P


nf4 = jnp.asarray(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)

def nf4xf32_to_f32(x):
    x = x.astype(jnp.float32)
    return (
        x
        * (
            x
            * (
                x
                * (
                    x * (1.82943132356953e-5 * x - 0.00068587779130373)
                    + 0.0100420261313669
                )
                - 0.0722703570217226
            )
            + 0.346075459755188
        )
        - 0.994166218659335
    )


def nf4xf32_to_f32_eqmul(x):
    x = x.astype(jnp.float32)
    result = jnp.zeros_like(x)
    for i, c in enumerate(nf4.tolist()):
        result = jnp.where(x == i, c, result)
    return result

def nf4xf32_to_f32_select(x):
    x = x.astype(jnp.int32)
    options = [
        jnp.full(fill_value=c, dtype=jnp.float32, shape=x.shape) for c in nf4.tolist()
    ]
    # options = nf4.tolist()
    for level in range(4):
        step = 2 ** (level)
        a = options[::2]
        b = options[1::2]
        new_options = []
        for k, (i, j) in enumerate(zip(a, b)):
            new_options.append(jnp.where(x < k * step * 2 + step, i, j))
        options = new_options
    return options[0]


sr = jax.lax.shift_right_logical
sl = jax.lax.shift_left
ba = jax.lax.bitwise_and

def i8tou8(x):
    return jnp.where(x < 0, 256 + x, x)


def i4tou4(x):
    return jnp.where(x < 0, 16 + x, x)


@partial(jax.jit, static_argnames=("kernel", "backward", "blocks"))
def matmul_fast(inputs, *tensors, kernel, backward=False, blocks=None):
    weight_transpose = backward

    inputs = inputs.astype(jnp.bfloat16)
    tensors = [
        t if t.dtype.kind not in ("V", "f") else t.astype(jnp.bfloat16) for t in tensors
    ]
    # tensors = [t.view(jnp.int8) if t.dtype == jnp.uint8 else t for t in tensors]

    if blocks is None:
        if not backward:
            # block_x, block_y, block_k = 4096, 256, 256  # 78%
            block_x, block_y, block_k = 2048, 512, 512 # 82.9%
        else:
            # block_x, block_y, block_k = 256, 1024, 256
            block_x, block_y, block_k = 256, 256, 512
    else:
        block_x, block_y, block_k = blocks

    if not weight_transpose:
        # tensor 0 is special and is fullest
        y = tensors[0].shape[2]
        quant_group_size = tensors[0].shape[1]
    else:
        quant_group_size = tensors[0].shape[1]
        y = quant_group_size * tensors[0].shape[0]

    x = inputs.shape[0]
    k = inputs.shape[1]
    if x < block_x:
        block_x = max(16, int(2 ** np.floor(np.log2(x))))
    if y < block_y:
        # block_y = max(16 if not backward else 1024, int(2 ** np.floor(np.log2(per_mp_output_size))))
        block_y = max(16, int(2 ** np.floor(np.log2(y))))
    if k < block_k:
        block_k = max(128, int(2 ** np.floor(np.log2(k))))
    x_pad = (block_x - x) % block_x
    k_pad = (block_k - k) % block_k
    if x_pad or k_pad:
        inputs = jnp.pad(
            inputs.reshape(inputs.shape[0] // x, x, -1, k),
            ((0, 0), (0, x_pad), (0, 0), (0, k_pad)),
        )
        inputs = inputs.reshape(-1, *inputs.shape[-2:])
        inputs = inputs.reshape(inputs.shape[0], -1)

    y_pad = (block_y - y) % block_y
    if y_pad:
        if not weight_transpose:
            tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, y_pad))) for t in tensors]
        else:
            tensors = [
                jnp.pad(t, ((0, y_pad // quant_group_size), (0, 0), (0, 0)))
                for t in tensors
            ]
    if k_pad:
        if not weight_transpose:
            tensors = [
                jnp.pad(t, ((0, k_pad // quant_group_size), (0, 0), (0, 0)))
                for t in tensors
            ]
        else:
            tensors = [jnp.pad(t, ((0, 0), (0, 0), (0, k_pad))) for t in tensors]
    if not weight_transpose:
        assert inputs.shape[1] == tensors[0].shape[0] * quant_group_size
    else:
        assert inputs.shape[1] == tensors[0].shape[2]

    def kernel_call(inputs, *tensors):
        inputs_dtype = inputs.dtype
        grid = ceil(inputs.shape[0] / block_x), ceil(y / block_y), ceil(k / block_k)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(lambda i, j, k: (i, k), (block_x, block_k)),
            ]
            + [
                pl.BlockSpec(
                    lambda i, j, k: (k, 0, j),
                    (block_k // quant_group_size, t.shape[1], block_y),
                )
                if not weight_transpose
                else pl.BlockSpec(
                    lambda i, j, k: (j, 0, k),
                    (block_y // quant_group_size, t.shape[1], block_k),
                )
                for t in tensors
            ],
            out_specs=pl.BlockSpec(lambda i, j, k: (i, j), (block_x, block_y)),
            scratch_shapes=[pltpu.VMEM((block_x, block_y), jnp.float32)],
        )
        outputs = pl.pallas_call(
            partial(kernel, block_k=block_k),
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct(
                (grid[0] * block_x, grid[1] * block_y), inputs_dtype
            ),
            compiler_params=dict(
                mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
            ),
            interpret=False,
        )(inputs, *tensors)
        if backward:
            outputs = outputs.reshape(outputs.shape[0], -1, 2, block_y // 2).mT.reshape(
                outputs.shape
            )
        return outputs

    # TODO shmap
    result = kernel_call(inputs, *tensors)
    if x_pad or y_pad:
        result = result[:x, :y]
    return result


def bf16_to_f32(x):
    sign = sr(x, 15)
    exp = sr(x, 7) & 0xFF
    man = x & 0x3FF

    sign = 1 - sign * 2
    exp = jnp.exp2(exp - 127)
    # man = exp + exp * (man / 1024.0)
    man = exp * (1 + man / 1024.0)

    return sign * man


int32_mask = int(jnp.array(0xFFFF0000, dtype=jnp.uint32).view(jnp.int32))
def matmul_nf4_kernel(
    inputs_ref, quants_ref, scale_ref, outputs_ref, accum_ref,
    *, block_k
):
    @pl.when(pl.program_id(axis=2) == 0)
    def _():
        accum_ref[...] = jnp.zeros_like(accum_ref)


    quants = quants_ref[...]
    scale = scale_ref[...]

    quants = i4tou4(quants.astype(jnp.int32))
    w1 = nf4xf32_to_f32(quants) * scale
    inputs = inputs_ref[...]
    accum_ref[...] += jnp.dot(inputs, w1.reshape(block_k, -1), preferred_element_type=jnp.float32)
    
    @pl.when(pl.program_id(axis=2) == (pl.num_programs(axis=2) - 1))
    def _():
        outputs_ref[...] = accum_ref[...].astype(outputs_ref.dtype)


def matmul_nf4_kernel_transpose(
    inputs_ref,
    quants_ref,
    scale_ref,
    outputs_ref,
    # accum1_ref, accum2_ref,
    accum_ref,
    *,
    block_k,
):
    accum_ref[...] = jnp.zeros_like(accum_ref)

    loop_iterations = max(1, inputs_ref.shape[-1] // block_k)

    def matmul_loop(i, _):
        inputs = pl.load(inputs_ref, (slice(None), pl.dslice(i * block_k, block_k)))
        quants = pl.load(
            quants_ref, (slice(None), slice(None), pl.dslice(i * block_k, block_k))
        )
        scale = pl.load(
            scale_ref, (slice(None), slice(None), pl.dslice(i * block_k, block_k))
        )

        to_nf4 = nf4xf32_to_f32

        quants = quants.astype(jnp.int32)
        quants = i8tou8(quants)
        # within 1 byte
        w1 = to_nf4(sr(quants, 4)) * scale
        w1 = w1.reshape(-1, inputs.shape[-1])
        w2 = to_nf4(quants & 0b1111) * scale
        w2 = w2.reshape(-1, inputs.shape[-1])

        output1 = inputs @ w1.T
        output2 = inputs @ w2.T

        output = jnp.concatenate((output1, output2), -1)
        accum_ref[...] += output

    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None)
    accum = accum_ref[...]
    outputs_ref[...] = accum.astype(outputs_ref.dtype)


def dequantize_group(group, scale, codebook, orig_dtype):
    group = i4tou4(group.astype(np.int32))

    codebook = codebook.astype(orig_dtype)

    stacked = nf4xf32_to_f32_select(group)

    return (stacked * scale).reshape(-1)

def dequantize(quants, scales):
    codebook = nf4

    half_group_size = quants.shape[1]

    quants = quants.astype(np.int32)
    grouped = quants.transpose(2, 0, 1).reshape(-1, half_group_size)
    scales = scales.transpose(2, 0, 1).reshape(-1)

    dequantized_groups = jax.vmap(dequantize_group, in_axes=(0, 0, None, None))(
        grouped, scales, codebook, jnp.bfloat16
    )

    unquant_matrix = dequantized_groups.reshape(quants.shape[2], -1).T

    return unquant_matrix

def time_mfu(number=100, blocks=None, verbose=False):
    import timeit
    if verbose:
        timeit.template = """
def inner(_it, _timer{init}):
    from tqdm import tqdm
    {setup}
    _t0 = _timer()
    for _i in tqdm(_it, total=_it.__length_hint__()):
        {stmt}
    _t1 = _timer()
    return _t1 - _t0
    """
    else:
        timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        {stmt}
    _t1 = _timer()
    return _t1 - _t0
    """
    matmul_fast(inputs, quants, scale, kernel=matmul_nf4_kernel, blocks=blocks).block_until_ready()
    time_per_iter = timeit.timeit(lambda: matmul_fast(inputs, quants, scale, kernel=matmul_nf4_kernel, blocks=blocks).block_until_ready(),
                                  number=number,) / number
    max_flops = 275e12
    flops_in_matmul = a * b * c * 2
    mfu = flops_in_matmul / (time_per_iter * max_flops)
    return mfu

if __name__ == "__main__":
    # x = jnp.arange(16, dtype=jnp.int32)
    # print(nf4[x].tolist())
    # print(nf4xf32_to_f32(x).tolist())
    # print((nf4xf32_to_f32_eqmul(x) - nf4).tolist())
    # print((nf4xf32_to_f32_select(x) - nf4).tolist())
    # exit()
    a, b, c, bs = 8192, 8192, 16384, 64
    # a, b, c, bs = 2048, 2048, 2048, 64
    quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs, b), -128, 127, dtype=jnp.int8).astype(jnp.int4)
    scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
    outputs = matmul_fast(inputs, quants, scale, kernel=matmul_nf4_kernel)
    outputs_ = inputs @ dequantize(quants, scale)
    print(jnp.mean(jnp.abs(outputs - outputs_)), jnp.mean(jnp.abs(outputs_)))
    mfu = time_mfu(100, verbose=True)
    print(f"MFU: {mfu:.2f}")

    exit()

    options = [256, 512, 1024, 2048, 4096]
    import random
    from tqdm.auto import tqdm
    from itertools import product
    options = list(product(options, options, options))
    random.shuffle(options)
    max_mfu = -1
    best = (0, 0, 0)
    failed = set()
    for x, y, z in (bar := tqdm(options)):
        group = (x, y, z)
        for f in failed:
            if all(f[i] < group[i] for i in range(3)):
                continue
        try:
            mfu = time_mfu(10, group)
        except jaxlib.xla_extension.XlaRuntimeError:
            failed.add(group)
            mfu = 0
        max_mfu, best = max((max_mfu, best), (mfu, group))
        bar.set_postfix(max_mfu=max_mfu, best=best)
    print(f"Best MFU: {max_mfu:.2f} with {best}")
    derivative = outputs / outputs.max()
    backwards = matmul_fast(derivative, quants, scale, kernel=matmul_nf4_kernel_transpose, backward=True)
