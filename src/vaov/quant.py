import jax
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


approx_nf4 = nf4xf32_to_f32(jnp.arange(16))
sr = jax.lax.shift_right_logical

def i8tou8(x):
    return jnp.where(x < 0, 256 + x, x)


@partial(jax.jit, static_argnames=("kernel", "backward"))
def matmul_fast(inputs, *tensors, kernel, backward=False):
    weight_transpose = backward
    inputs_32 = not backward

    inputs = inputs.astype(jnp.bfloat16)
    tensors = [
        t if t.dtype.kind not in ("V", "f") else t.astype(jnp.bfloat16) for t in tensors
    ]
    # tensors = [t.view(jnp.int8) if t.dtype == jnp.uint8 else t for t in tensors]

    if not backward:
        block_x, block_y, block_k = 256, 256, 512
    else:
        # block_x, block_y, block_k = 256, 1024, 256
        block_x, block_y, block_k = 256, 256, 512

    if not weight_transpose:
        # tensor 0 is special and is fullest
        y = tensors[0].shape[2]
        quant_group_size = tensors[0].shape[1] * 2
    else:
        quant_group_size = tensors[0].shape[1] * 2
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
        if inputs_32:
            inputs = inputs.view(jnp.int32)
        grid = ceil(inputs.shape[0] / block_x), ceil(y / block_y)
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(lambda i, j: (i, 0), (block_x, inputs.shape[1])),
            ]
            + [
                pl.BlockSpec(lambda i, j: (0, 0, j), (t.shape[0], t.shape[1], block_y))
                if not weight_transpose
                else pl.BlockSpec(
                    lambda i, j: (j, 0, 0),
                    (block_y // quant_group_size, t.shape[1], t.shape[2]),
                )
                for t in tensors
            ],
            out_specs=pl.BlockSpec(lambda i, j: (i, j), (block_x, block_y)),
            scratch_shapes=[pltpu.VMEM((block_x, block_y), jnp.float32)],
        )
        outputs = pl.pallas_call(
            partial(kernel, block_k=block_k),
            grid_spec=grid_spec,
            out_shape=jax.ShapeDtypeStruct(
                (grid[0] * block_x, grid[1] * block_y), inputs_dtype
            ),
            compiler_params=dict(
                mosaic=dict(dimension_semantics=("parallel", "parallel"))
            ),
            # interpret=True,
        )(inputs, *tensors)
        if backward:
            print(outputs.reshape(outputs.shape[0], -1, 2, block_y // 2).mT.shape)
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


def matmul_nf4_kernel(
    inputs_ref, quants_ref, scale_ref, outputs_ref, accum_ref, *, block_k
):
    block_m = quants_ref.shape[2]
    quant_group_size = quants_ref.shape[1] * 2

    accum_ref[...] = jnp.zeros_like(accum_ref)

    block_group = block_k // quant_group_size
    loop_iterations = max(1, inputs_ref.shape[-1] * 2 // block_k)

    def matmul_loop(iteration, _):
        inputs = pl.load(
            inputs_ref,
            (slice(None), pl.dslice((iteration * (block_k // 2)), block_k // 2)),
        )

        quants = pl.load(
            quants_ref,
            (pl.dslice(iteration * block_group, block_group), slice(None), slice(None)),
        )
        scale = pl.load(
            scale_ref,
            (pl.dslice(iteration * block_group, block_group), slice(None), slice(None)),
        )

        # to_nf4 = lambda x: nf4[x]
        to_nf4 = nf4xf32_to_f32
        assert quants.dtype == jnp.int8

        sl = jax.lax.shift_left
        sr = jax.lax.shift_right_logical
        i8tou8 = lambda x: jnp.where(x < 0, 256 + x, x)

        quants = quants.astype(jnp.int32)
        quants = i8tou8(quants)
        # within 1 byte
        w1 = to_nf4(sr(quants, 4)) * scale
        w2 = to_nf4(quants & 0b1111) * scale

        # i = inputs.shape[-1] // 2
        i = block_k // 2
        # inputs_ = inputs.view(jnp.int32)

        # little-endian!
        inputs1 = inputs & 0xFFFF
        # inputs1 = sl(inputs1, 16).view(jnp.float32)
        inputs1 = bf16_to_f32(inputs1)
        inputs2 = sr(inputs, 16)
        # inputs2 = sl(inputs2, 16).view(jnp.float32)
        inputs2 = bf16_to_f32(inputs2)

        # inputs1 = pl.load(inputs_ref, (slice(None), pl.dslice(iteration*block_k, block_k // 2, 2)))
        # inputs2 = pl.load(inputs_ref, (slice(None), pl.dslice(iteration*block_k+1, block_k // 2, 2)))
        accum_ref[...] += inputs1 @ w1.reshape(i, -1)
        accum_ref[...] += inputs2 @ w2.reshape(i, -1)

    jax.lax.fori_loop(0, loop_iterations, matmul_loop, init_val=None)
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
        assert quants.dtype == jnp.int8

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


if __name__ == "__main__":
    import timeit
    a, b, c, bs = 4096, 4096, 4096, 32
    quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs // 2, b), 0, 255, dtype=jnp.int8)
    scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
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
    outputs = matmul_fast(inputs, quants, scale, kernel=matmul_nf4_kernel)
    number = 20
    time_per_iter = timeit.timeit(lambda: matmul_fast(inputs, quants, scale, kernel=matmul_nf4_kernel).block_until_ready(),
                                  number=number,) / number
    print(f"Time: {time_per_iter:.4f}")
    max_flops = 275e12
    flops_in_matmul = a * b * c * 2
    mfu = flops_in_matmul / (time_per_iter * max_flops)
    print(f"MFU: {mfu:.2f}")
    derivative = outputs / outputs.max()
    backwards = matmul_fast(derivative, quants, scale, kernel=matmul_nf4_kernel_transpose, backward=True)
