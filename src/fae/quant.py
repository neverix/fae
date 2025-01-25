from typing import Optional, Tuple, Sequence, List
import jax
import equinox as eqx
from dataclasses import dataclass
import dataclasses
import jax.experimental
import jax.experimental.shard_map
import jax.numpy as jnp
import numpy as np
from math import ceil
from concurrent import futures
import orbax.checkpoint as ocp
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
from contextlib import contextmanager
import qax


USE_KERNEL = False

@contextmanager
def kernel_mode(value: bool):
    global USE_KERNEL
    old = USE_KERNEL
    USE_KERNEL = value
    yield
    USE_KERNEL = old

BIG_POLYNOMIAL = False
def nf4xf32_to_f32(x):
    x = x.astype(jnp.float32)
    if BIG_POLYNOMIAL:
        return (
            x
            * (
                x
                * (
                    x
                    * (
                        x
                        * (
                            x
                            * (
                                x
                                * (
                                    x
                                    * (
                                        x
                                        * (
                                            x
                                            * (
                                                6.88674345241901e-8
                                                - 8.47182255893858e-10 * x
                                            )
                                            - 2.38380978507793e-6
                                        )
                                        + 4.62161872192984e-5
                                    )
                                    - 0.000555886625699011
                                )
                                + 0.00435485724102389
                            )
                            - 0.0228024324616347
                        )
                        + 0.0811200908594962
                    )
                    - 0.200455927752031
                )
                + 0.441938722753289
            )
            - 0.999980040956333
        )
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


with jax.default_device(jax.devices("cpu")[0]):
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
    approx_nf4 = nf4xf32_to_f32(jnp.arange(16))


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


def u4toi4(x):
    return jnp.where(x >= 8, x - 16, x)


def i4tou4(x):
    return jnp.where(x < 0, 16 + x, x)


BLOCK_OVERRIDE = None
@partial(jax.jit, static_argnames=("kernel", "backward", "blocks"))
def matmul_fast(inputs, *tensors, kernel, backward=False, blocks=None):
    weight_transpose = backward

    inputs = inputs.astype(jnp.bfloat16)
    # tensors = [t if t.dtype.kind not in ("V", "f") else t.astype(jnp.bfloat16) for t in tensors]
    # tensors = [t.view(jnp.int8) if t.dtype == jnp.uint8 else t for t in tensors]

    if blocks is None:
        if not backward:
            if BLOCK_OVERRIDE is not None:
                block_x, block_y, block_k = BLOCK_OVERRIDE
            else:
                # block_x, block_y, block_k = 2048, 512, 512  # 78%
                block_x, block_y, block_k = 2048, 512, 256
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
            inputs.reshape(x, k),
            ((0, x_pad), (0, k_pad)),
        )

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
            interpret=jax.devices()[0].platform == "cpu",
        )(inputs, *tensors)
        if backward:
            outputs = outputs.reshape(outputs.shape[0], -1, 2, block_y // 2).mT.reshape(
                outputs.shape
            )
        return outputs

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

    if quants.dtype == jnp.int8:
        w1 = (quants.astype(jnp.float32) / 127.5) * scale.astype(jnp.float32)
    else:
        quants = i4tou4(quants.astype(jnp.int32))
        quants = nf4xf32_to_f32(quants)
        # quants = quants.astype(jnp.float32)
        w1 = quants * scale
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


@partial(jax.jit, static_argnums=(1, ))
def dequantize_quants(group, mode):
    if mode == "nf4":
        group = i4tou4(group.astype(jnp.int32))
        stacked = nf4xf32_to_f32(group)
    else:
        stacked = group.astype(jnp.float32) / 127.5

    return stacked

@partial(jax.jit, static_argnums=(3, 4))
def dequantize_group(group, scale, codebook, orig_dtype, mode):

    # codebook = codebook.astype(orig_dtype)
    stacked = dequantize_quants(group, mode)

    return (stacked * scale).reshape(-1).astype(orig_dtype)


def dequantize_vmap(quants, scales, *, use_approx, orig_dtype, mode):
    if quants.ndim > 3:
        return jax.vmap(
            partial(dequantize_vmap, use_approx=use_approx, orig_dtype=orig_dtype, mode=mode),
            in_axes=(0, 0),
            out_axes=0)(quants, scales)

    codebook = approx_nf4 if use_approx else nf4

    group_size = quants.shape[1]

    quants_deq = dequantize_quants(quants, mode)
    full_weights = quants_deq * scales
    return full_weights.reshape(-1, quants.shape[-1]).astype(orig_dtype)

    # unquant_matrix = jax.vmap(
    #     jax.vmap(dequantize_group, in_axes=(1, 1, None, None, None)),
    #     in_axes=(0, 0, None, None, None),
    # )(quants, scales, codebook, orig_dtype, mode)
    # unquant_matrix = unquant_matrix.reshape(-1, quants.shape[2])

    grouped = quants.transpose(2, 0, 1).reshape(-1, group_size)
    # grouped = quants.reshape(-1, group_size)
    scales = scales.transpose(2, 0, 1).reshape(-1)
    # scales = scales.reshape(-1)

    dequantized_groups = jax.vmap(dequantize_group, in_axes=(0, 0, None, None, None))(
        grouped, scales, codebook, orig_dtype, mode
    )

    unquant_matrix = dequantized_groups.reshape(quants.shape[2], -1).T

    return unquant_matrix


@dataclass
class QuantMatrix(qax.ImplicitArray, warn_on_materialize=True):
    quants: jnp.ndarray
    scales: jnp.ndarray

    use_approx: bool = qax.aux_field()
    use_kernel: bool = qax.aux_field()
    orig_dtype: jnp.dtype = qax.aux_field()

    mesh_and_axis: Optional[Tuple[jax.sharding.Mesh, Optional[int]]] = qax.aux_field()

    @staticmethod
    def quantize(mat, mode="nf4", **kwargs):
        quants, scales = quantize_vmap(mat, mode, **kwargs)
        return QuantMatrix(
            quants=quants,
            scales=scales,
            use_approx=True,
            orig_dtype=mat.dtype,
            use_kernel=True,
            mesh_and_axis=None,
        )

    def __post_init__(self):
        self.dtype = self.compute_dtype()
        self.shape = self.compute_shape()

    def compute_dtype(self):
        return self.orig_dtype

    def compute_shape(self):
        return self.quants.shape[:-3] + (
            self.quants.shape[-3] * self.quants.shape[-2],
            self.quants.shape[-1],
        )

    @property
    def block_size(self):
        return self.quants.shape[-2]

    def slice(self, *, axis: int, start: int, size: int):
        if axis == self.quants.ndim - 3:
            start //= self.block_size
            size //= self.block_size
        return dataclasses.replace(
            self,
            quants=jax.lax.dynamic_slice_in_dim(self.quants, start, size, axis),
            scales=jax.lax.dynamic_slice_in_dim(self.scales, start, size, axis))

    def stack(self, *quants):
        return QuantMatrix(
            quants=jnp.stack((self.quants,) + tuple(x.quants for x in quants), axis=0),
            scales=jnp.stack((self.scales,) + tuple(x.scales for x in quants), axis=0),
            use_approx=self.use_approx, use_kernel=self.use_kernel, orig_dtype=self.orig_dtype,
            mesh_and_axis=self.mesh_and_axis
        )


    def materialize(self):
        if self.use_kernel:
            # We should never materialize if we're trying to use the kernel
            raise NotImplementedError

        return self.dequantize()

    @property
    def mode(self):
        if self.quants.dtype == jnp.int8:
            return "i8"
        elif self.quants.dtype == jnp.int4:
            return "nf4"

    def dequantize(self):
        return dequantize_vmap(self.quants, self.scales,
                               use_approx=self.use_approx, orig_dtype=self.orig_dtype,
                               mode=self.mode)

    # @partial(jax.jit, static_argnames=("mesh_and_axis"))
    def with_mesh_and_axis(self, mesh_and_axis):
        batch_dims = (None,) * (self.quants.ndim - 3)
        new_quant_matrix = dataclasses.replace(self, mesh_and_axis=mesh_and_axis)
        mesh, shard_axis = mesh_and_axis
        if shard_axis == 0:
            quant_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*batch_dims, "tp", None, None)
            )
        elif shard_axis == 1:
            quant_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*batch_dims, None, None, "tp")
            )
        elif shard_axis is None:
            quant_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*batch_dims, None, None, "fsdp")
            )
        new_quant_matrix = jax.device_put(new_quant_matrix, quant_sharding)
        return new_quant_matrix

    # @classmethod
    # def default_handler(cls, primitive, *args, params=None):
    #     if params is None:
    #         params = {}
    #     # print("aaaa", type(primitive), primitive, [str(a)[:50] for a in args], {
    #     #     k: str(v)[:50] for k, v in params.items()
    #     # })
    # #     # print("aa", primitive, args, params)
    # #     # exit()
    #     subfuns, bind_params = primitive.get_bind_params(params)
    #     # return qax.use_implicit_args(primitive.bind)(*subfuns, *args, **bind_params)
    #     return primitive.bind(*subfuns, *args, **bind_params)
    #     # return materialize_handler(primitive, *args, params=params)


@qax.primitive_handler("slice")
def slice_handler(
    primitive, a: QuantMatrix, *, start_indices, limit_indices, **kwargs
):
    start_indices = start_indices[:-2] + (0, 0, 0)
    limit_indices_quants = limit_indices[:-2] + a.quants.shape[-3:]
    limit_indices_scales = limit_indices[:-2] + a.scales.shape[-3:]
    return dataclasses.replace(
        a,
        quants=jax.lax.slice(a.quants, start_indices=start_indices, limit_indices=limit_indices_quants),
        scales=jax.lax.slice(a.scales, start_indices=start_indices, limit_indices=limit_indices_scales),
    )

@qax.primitive_handler("squeeze")
def squeeze_handler(
    primitive, a: QuantMatrix, *, dimensions, **kwargs
):
    for axis in dimensions:
        assert 0 <= axis < (a.quants.ndim - 3)
    return dataclasses.replace(
        a,
        quants=jax.lax.squeeze(a.quants, dimensions),
        scales=jax.lax.squeeze(a.scales, dimensions),
    )

def dot_general_handler(
    primitive, a: jax.Array, b: QuantMatrix, *, dimension_numbers, **kwargs
):
    if not b.use_kernel:
        return NotImplemented

    while a.shape[-1] != b.shape[0]:
        a = a.reshape(a.shape[:-2], -1)
        if a.shape[-1] > b.shape[0]:
            raise NotImplementedError

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if (
        rhs_batch
        or lhs_batch
        or lhs_contract != (len(a.shape) - 1,)
        or rhs_contract != (0,)
    ):
        # Fall back to materialization
        return NotImplemented

    if not b.use_approx:
        # No kernel for NF4
        return NotImplemented

    og_dtype = a.dtype
    compute_dtype = jnp.bfloat16  # just so there's no confusion
    a = a.astype(compute_dtype)

    orig_a_shape = a.shape
    if USE_KERNEL:
        mf = partial(matmul_fast, kernel=matmul_nf4_kernel)
    else:
        def dq(*tensors):
            return dequantize_vmap(*tensors,
                                    use_approx=b.use_approx,
                                    orig_dtype=og_dtype,
                                    mode="i8" if b.quants.dtype == jnp.int8 else "nf4")

        @partial(jax.custom_vjp)
        def mf(inputs, *tensors):
            return (inputs @ dq(*tensors)).astype(compute_dtype)

        def mf_fw(inputs, *tensors):
            m = dq(*tensors)
            return (inputs @ m).astype(compute_dtype), (m,)

        def mf_bw(t, grads):
            m, = t
            return (grads @ m.T).astype(compute_dtype), None, None

        mf.defvjp(mf_fw, mf_bw)
        mf = jax.remat(mf)

    if b.mesh_and_axis is not None:
        mesh, map_axis = b.mesh_and_axis
        tensors = b.quants, b.scales
        # Reshape a to be 3-D
        # (dp, fsdp, tp)
        a = a.reshape(-1, a.shape[1], a.shape[-1])
        if map_axis in (0, 1):
            def matmul(inputs, *tensors):
                orig_inputs_shape = inputs.shape
                inputs = inputs.reshape(-1, inputs.shape[-1])
                outputs = mf(inputs, *tensors)
                outputs = outputs.reshape(*orig_inputs_shape[:-1], outputs.shape[-1])
                if map_axis == 0:
                    outputs = jax.lax.psum(outputs, axis_name="tp")
                return outputs
            out = jax.experimental.shard_map.shard_map(
                matmul,
                mesh=mesh,
                in_specs=(
                    P("dp", "fsdp", "tp" if map_axis == 0 else None),
                ) + tuple(P("tp" if map_axis == 0 else None, None, "tp" if map_axis == 1 else None) for _ in tensors),
                out_specs=P("dp", "fsdp", "tp" if map_axis == 1 else None),
                check_rep=False,  # No replication rule for pallas_call
            )(a, *tensors)
        # map_axis == None
        elif mesh.shape["fsdp"] == 1:
            def matmul_inner(a, *tensors):
                a = a.reshape(-1, a.shape[-1])
                out = mf(a, *tensors)
                return out
            a = a.reshape(-1, a.shape[1], a.shape[-1])
            out = jax.experimental.shard_map.shard_map(
                matmul_inner,
                mesh=mesh,
                in_specs=(P("dp", "fsdp", None),)
                + tuple(
                    P(
                        None,
                        None,
                        None,
                    )
                    for _ in tensors
                ),
                out_specs=P("dp", "fsdp", None),
                check_rep=False,  # No replication rule for pallas_call
            )(
                a,
                *tensors,
            )
        else:
            # https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html#a-realistic-transformer-example
            def matmul(inputs, *tensors):
                axis_size = jax.lax.psum(1, axis_name="fsdp")
                axis_index = jax.lax.axis_index(axis_name="fsdp")

                accum = jnp.zeros((inputs.shape[0], inputs.shape[1], b.shape[1]), dtype=compute_dtype)
                def loop_body(i, args):
                    accum, inputs, tensors = args
                    partial_result = mf(
                        inputs.reshape(-1, inputs.shape[-1]),
                        *tensors,
                    )
                    partial_result = partial_result.reshape(*inputs.shape[:-1], -1)
                    chunk_size = partial_result.shape[-1]
                    accum = jax.lax.dynamic_update_slice(accum, partial_result, (0, 0, ((axis_index + i) % axis_size)*chunk_size))
                    # inputs, tensors = jax.lax.ppermute(
                    #     (inputs, tensors),
                    #     axis_name="fsdp",
                    #     perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
                    # )
                    tensors = jax.lax.ppermute(
                        tensors,
                        axis_name="fsdp",
                        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
                    )
                    return accum, inputs, tensors
                accum, inputs, tensors = jax.lax.fori_loop(0, axis_size - 1, loop_body, (accum, inputs, tensors))
                partial_result = mf(inputs.reshape(-1, inputs.shape[-1]), *tensors)
                partial_result = partial_result.reshape(*inputs.shape[:-1], -1)
                chunk_size = partial_result.shape[-1]
                i = axis_size - 1
                accum = jax.lax.dynamic_update_slice(accum, partial_result, (0, 0, ((axis_index + i) % axis_size)*chunk_size))
                outputs = accum

                return outputs
            out = jax.experimental.shard_map.shard_map(
                matmul,
                mesh=mesh,
                in_specs=(
                    P("dp", "fsdp", None),
                ) + tuple(P(None, None, "fsdp") for _ in tensors),
                out_specs=P("dp", "fsdp", None),
                check_rep=False,  # No replication rule for pallas_call
            )(a, *tensors)
    else:
        # Reshape a to be 2-D
        a = a.reshape(-1, b.shape[0])
        out = mf(a, b.quants, b.scales)
    return out.reshape(*orig_a_shape[:-1], out.shape[-1]).astype(og_dtype)


qax.primitive_handler("dot_general")(dot_general_handler)


class MockQuantMatrix(eqx.Module):
    quants: jnp.ndarray
    scales: jnp.ndarray

    use_approx: bool = eqx.field(static=True)
    use_kernel: bool = eqx.field(static=True)
    orig_dtype: jnp.dtype = eqx.field(static=True)

    mesh_and_axis: Optional[Tuple[jax.sharding.Mesh, Optional[int]]] = eqx.field(static=True)

    def __repr__(self):
        return f"MockQuantMatrix(quants={self.quants.shape}, scales={self.scales.shape}, use_approx={self.use_approx}, use_kernel={self.use_kernel}, orig_dtype={self.orig_dtype}, mesh_and_axis=something)"

    @classmethod
    def mockify(cls, pytree):
        def _mockify(x):
            if not isinstance(x, QuantMatrix):
                return x
            return MockQuantMatrix(**{k: getattr(x, k) for k in {"quants", "scales", "use_approx", "use_kernel", "orig_dtype", "mesh_and_axis"}})
        return jax.tree.map(_mockify, pytree, is_leaf=is_arr)

    @classmethod
    def unmockify(cls, pytree):
        def _unmockify(x):
            if not isinstance(x, MockQuantMatrix):
                return x
            return QuantMatrix(**{k: getattr(x, k) for k in {"quants", "scales", "use_approx", "use_kernel", "orig_dtype", "mesh_and_axis"}})
        return jax.tree.map(_unmockify, pytree, is_leaf=lambda x: is_arr(x) or isinstance(x, MockQuantMatrix))

    @property
    def shape(self):
        return self.quants.shape[:-3] + (
            self.quants.shape[-3] * self.quants.shape[-2],
            self.quants.shape[-1],
        )

    @property
    def dtype(self):
        return self.orig_dtype

    @staticmethod
    def quantize(mat, mode="nf4", **kwargs):
        quants, scales = quantize_vmap(mat, mode, **kwargs)
        return MockQuantMatrix(
            quants=quants,
            scales=scales,
            use_approx=True,
            orig_dtype=mat.dtype,
            use_kernel=True,
            mesh_and_axis=None,
        )

    def slice(self, *, axis: int, start: int, size: int):
        if axis == self.quants.ndim - 3:
            start //= self.block_size
            size //= self.block_size
        return dataclasses.replace(
            self,
            quants=jax.lax.dynamic_slice_in_dim(self.quants, start, size, axis),
            scales=jax.lax.dynamic_slice_in_dim(self.scales, start, size, axis),
        )

    def stack(self, *quants):
        return dataclasses.replace(
            self,
            quants=jnp.stack((self.quants,) + tuple(x.quants for x in quants), axis=0),
            scales=jnp.stack((self.scales,) + tuple(x.scales for x in quants), axis=0),
        )
    
    @property
    def ndim(self):
        return self.quants.ndim - 1

    def with_mesh_and_axis(self, mesh_and_axis):
        batch_dims = (None,) * (self.quants.ndim - 3)
        new_quant_matrix = dataclasses.replace(self, mesh_and_axis=mesh_and_axis)
        mesh, shard_axis = mesh_and_axis
        if shard_axis == 0:
            quant_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*batch_dims, "tp", None, None)
            )
        elif shard_axis == 1:
            quant_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*batch_dims, None, None, "tp")
            )
        elif shard_axis is None:
            quant_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*batch_dims, None, None, "fsdp")
            )
        # new_quant_matrix = jax.device_put(
        #     new_quant_matrix,
        #     MockQuantMatrix(quants=quant_sharding, scales=quant_sharding))
        new_quant_matrix = dataclasses.replace(
            new_quant_matrix,
            quants=jax.device_put(new_quant_matrix.quants, quant_sharding),
            scales=jax.device_put(new_quant_matrix.scales, quant_sharding),
        )
        return new_quant_matrix

def is_arr(x):
    return isinstance(x, (MockQuantMatrix, qax.primitives.ArrayValue))

# # https://github.com/jax-ml/jax/blob/main/jax/experimental/jet.py
# def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees, *args, **kwargs):
#     del primitive, fwd, bwd, out_trees  # Unused.
#     return fun.call_wrapped(*tracers)
# def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
#                               symbolic_zeros):
#     del primitive, jvp  # Unused.
#     return fun.call_wrapped(*tracers)
# from qax.implicit.implicit_array import ImplicitArrayTrace
# ImplicitArrayTrace.process_custom_vjp_call = process_custom_vjp_call
# ImplicitArrayTrace.process_custom_jvp_call = process_custom_jvp_call

def quantize_groups(group, codebook, mode="nf4"):
    group = group.astype(jnp.float32)

    scale = jnp.max(jnp.abs(group))
    scaled = group / scale

    if mode == "nf4":
        errors = scaled[..., None] - codebook
        quants = jnp.argmin(jnp.abs(errors), axis=-1)
    elif mode == "i8":
        scaled = scaled * 127.5 - 0.5
        quants = jnp.clip(scaled.round(), -128, 127).astype(jnp.int8)

    return quants, scale

@partial(jax.jit, static_argnames=("use_approx", "group_size", "mesh_and_axis", "quantize_groups", "codebook", "mode"))
def quantize_matrix(mat, use_approx, group_size=32, mesh_and_axis=None, quantize_groups=quantize_groups, codebook=approx_nf4, mode="nf4"):
    transposed = mat.T
    grouped = transposed.reshape(-1, group_size)

    quants, scales = jax.vmap(partial(quantize_groups, mode=mode), in_axes=(0, None))(grouped, codebook)

    # int4 does not support reshape/transpose on CPU
    quants = quants.reshape(mat.shape[-1], -1, group_size)
    quants = quants.transpose(1, 2, 0)

    if mode == "nf4":
        quants = u4toi4(quants)
        quants = quants.astype(jnp.int4)

    scales = scales.reshape(mat.shape[-1], -1, 1)
    scales = scales.transpose(1, 2, 0)

    return QuantMatrix(
        quants=quants,
        scales=scales,
        use_approx=use_approx,
        orig_dtype=mat.dtype,
        use_kernel=True,
        mesh_and_axis=mesh_and_axis,
    )


@partial(jax.jit, static_argnames=("mode", "group_size", "mesh_and_axis"))
def quantize_vmap(mat, mode, group_size=32, mesh_and_axis=None):
    if mat.ndim > 2:
        return jax.vmap(
            partial(quantize_vmap, mode=mode, group_size=group_size, mesh_and_axis=mesh_and_axis)
        )(mat)
    mat = quantize_matrix(mat, use_approx=True, group_size=group_size, mesh_and_axis=mesh_and_axis, mode=mode)
    return mat.quants, mat.scales

def time_mfu(number=100, blocks=None, verbose=False):
    global BLOCK_OVERRIDE
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
    # clearer than jax.jit(qax.use_implicit_args(jnp.dot))
    @jax.jit
    @qax.use_implicit_args
    def computation_jit(inputs, quant_matrix):
        return inputs @ quant_matrix
    def computation():
        return computation_jit(inputs_device, quant_matrix).block_until_ready()
    BLOCK_OVERRIDE = blocks
    result = computation()
    time_per_iter = timeit.timeit(computation, number=number,) / number
    max_flops = 275e12 * len(jax.devices("tpu"))
    flops_in_matmul = a * b * c * 2
    mfu = flops_in_matmul / (time_per_iter * max_flops)
    BLOCK_OVERRIDE = None
    return result, mfu


def check_accuracy(inputs, matrix, quant_matrix):
    jax.debug.print("{x}, {y}", x=jnp.mean(jnp.abs(matrix - quant_matrix.dequantize())), y=jnp.mean(jnp.abs(matrix)))
    result = inputs @ matrix
    result_ = inputs @ quant_matrix.dequantize()
    jax.debug.print("{x}, {y}", x=jnp.mean(jnp.abs(result - result_)), y=jnp.mean(jnp.abs(result_)))
    return result_

if __name__ == "__main__":
    a, b, c, bs = 32768, 2048, 8192, 64
    # a, b, c, bs = 2048, 2048, 2048, 64
    quants = jax.random.randint(jax.random.PRNGKey(0), (a // bs, bs, b), 0, 255, dtype=jnp.int8)
    quants = quants.astype(jnp.int4)
    scale = jax.random.normal(jax.random.PRNGKey(1), (a // bs, 1, b), dtype=jnp.bfloat16) / 255
    inputs = jax.random.normal(jax.random.PRNGKey(2), (c, a), dtype=jnp.bfloat16)
    outputs = matmul_fast(inputs, quants, scale, kernel=matmul_nf4_kernel)
    inputs_device = inputs
    quant_matrix = QuantMatrix(quants=quants, scales=scale,
                               use_approx=True, orig_dtype=jnp.bfloat16, use_kernel=True,
                               mesh_and_axis=None)
    # outputs_ = inputs @ dequantize(quants, scale)
    # print(jnp.mean(jnp.abs(outputs - outputs_)), jnp.mean(jnp.abs(outputs_)))
    mfu = time_mfu(100, verbose=True)[1]
    print(f"MFU: {mfu:.2f}")
    options = [256, 512, 1024, 2048]
    import random
    from tqdm.auto import tqdm
    from itertools import product
    options = list(product(options, options, options))
    random.shuffle(options)
    max_mfu = 0
    best = None
    failed = set()
    for x, y, z in (bar := tqdm(options)):
        group = (x, y, z)
        for f in failed:
            if all(f[i] < group[i] for i in range(3)):
                continue
        try:
            mfu = time_mfu(10, group)[1]
        except Exception as e:
            print(type(e))
            failed.add(group)
            mfu = 0
        max_mfu, best = max((max_mfu, best), (mfu, group))
        bar.set_postfix(max_mfu=max_mfu, best=best)
    print(f"Best MFU: {max_mfu:.2f} with {best}")
    exit()

    # with jax.default_device(jax.devices("cpu")[0]):
    #     a, b, c = 512, 512, 512

    #     inputs = jax.random.normal(jax.random.key(0), (a, b), dtype=jnp.bfloat16)
    #     matrix = jax.random.normal(jax.random.key(1), (b, c), dtype=jnp.bfloat16)

    #     quant_matrix = {"a": [quantize_matrix(matrix, use_approx=True)], "b": [quantize_matrix(matrix, use_approx=False)]}

    #     vals, treedef = jax.tree.flatten(
    #         quant_matrix, is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue)
    #     )
    #     treedef = jax.tree.unflatten(treedef, [jnp.zeros((1,)) for x in vals])
    #     new_vals = []
    #     shardify = lambda val: jax.device_put(val, jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0]))
    #     for val in vals:
    #         if isinstance(val, QuantMatrix):
    #             quants, scales = shardify((val.quants, val.scales))
    #             new_vals.append(("qm", quants, scales, val.use_approx, str(val.orig_dtype), val.use_kernel, val.mesh_and_axis))
    #         else:
    #             new_vals.append(shardify(val))
    #     vals = new_vals

    #     checkpointer = ocp.PyTreeCheckpointer()
    #     path = ocp.test_utils.erase_and_create_empty("somewhere/test")
    #     path = path.resolve()
    #     checkpointer.save(path / "struc", treedef)
    #     checkpointer.save(path / "aa", vals)

    #     checkpointer = ocp.PyTreeCheckpointer()
    #     treedef_meta = checkpointer.metadata(path / "struc")
    #     sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    #     restore_args = ocp.checkpoint_utils.construct_restore_args(
    #         treedef_meta,
    #         sharding_tree=jax.tree.map(lambda _: sharding, treedef_meta)
    #     )
    #     restored_treedef = checkpointer.restore(path / "struc",
    #                                             restore_args=restore_args)
    #     vals_meta = checkpointer.metadata(path / "aa")
    #     sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    #     restore_args = ocp.checkpoint_utils.construct_restore_args(
    #         vals_meta,
    #         sharding_tree=jax.tree.map(lambda _: sharding, vals_meta)
    #     )
    #     restored_vals = checkpointer.restore(path / "aa",
    #                                             restore_args=restore_args)

    #     _, restored_treedef = jax.tree_flatten(
    #         restored_treedef, is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue)
    #     )
    #     new_vals = []
    #     for val in restored_vals:
    #         if isinstance(val, list) and val[0] == "qm":
    #             quants, scales, use_approx, orig_dtype, use_kernel, mesh_and_axis = val[1:]
    #             new_vals.append(QuantMatrix(
    #                 quants=quants, scales=scales,
    #                 use_approx=use_approx, orig_dtype=orig_dtype,
    #                 use_kernel=use_kernel, mesh_and_axis=mesh_and_axis))
    #         else:
    #             new_vals.append(val)
    #     restored_quant_matrix = jax.tree_unflatten(restored_treedef, new_vals)
    #     print(jax.tree.map(lambda a, b: (a == b).all(), restored_quant_matrix, quant_matrix))
    # exit()

    # shard_axis = None
    # mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(2, -1, 1), ("dp", "fsdp", "tp"))
    # quant_matrix = quant_matrix.with_mesh_and_axis((mesh, shard_axis))
    # input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))
    # inputs_device = jax.device_put(inputs.reshape(-1, 64, inputs.shape[-1]), input_sharding)

    # inputs_device = jax.block_until_ready(inputs_device)
    # quant_matrix = jax.block_until_ready(quant_matrix)
    # super_quant = quant_matrix.stack(quant_matrix)

    # @jax.jit
    # @qax.use_implicit_args
    # def h(a, b):
    #     return a @ b
    # print(inputs_device.shape, quant_matrix.shape, h(inputs_device, quant_matrix).shape)
    # print(super_quant.shape)
    # jax.debug.inspect_array_sharding(super_quant.quants, callback=print)

    # @jax.jit
    # def g(x):
    #     return jnp.mean(jnp.abs(x))

    # @jax.jit
    # @qax.use_implicit_args
    # def f(a, b):
    #     def inner(a, b):
    #         print(a.shape, b.shape, (a @ b).shape)
    #         return a @ b, None
    #     print(b)
    #     y = jax.lax.scan(inner, a, b,)[0]
    #     return jnp.mean(jnp.abs(y))
    # print(f(inputs_device, super_quant))
    # print(g(h(h(inputs_device, quant_matrix), quant_matrix)))

    # exit()

    # x = jnp.arange(16, dtype=jnp.int32)
    # print(nf4[x].tolist())
    # print(nf4xf32_to_f32(x).tolist())
    # print((nf4xf32_to_f32_eqmul(x) - nf4).tolist())
    # print((nf4xf32_to_f32_select(x) - nf4).tolist())

    a, b, c = 2**16, 2**15, 2**13


    with jax.default_device(jax.devices("cpu")[0]):
        inputs = jax.random.normal(jax.random.key(0), (a, b), dtype=jnp.bfloat16)
        matrix = jax.random.normal(jax.random.key(1), (b, c), dtype=jnp.bfloat16)

    quant_matrix = QuantMatrix.quantize(matrix, mode="i8")
    result = check_accuracy(inputs, matrix, quant_matrix)
    shard_axis = 0
    for shard_axis in (None, 0, 1):
        # testing FSDP-like weight replication/sharding
        if shard_axis is not None:
            mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(2, 1, -1), ("dp", "fsdp", "tp"))
        else:
            mesh = jax.sharding.Mesh(np.array(jax.devices("tpu")).reshape(2, -1, 1), ("dp", "fsdp", "tp"))
        quant_matrix = quant_matrix.with_mesh_and_axis((mesh, shard_axis))
        if shard_axis == 0:
            input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", "tp"))
        else:
            input_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp", "fsdp", None))
        inputs_device = jax.device_put(inputs.reshape(-1, 64, inputs.shape[-1]), input_sharding)

        inputs_device = jax.block_until_ready(inputs_device)
        quant_matrix = jax.block_until_ready(quant_matrix)

        result_, mfu = time_mfu(100, verbose=True)
        print(f"Shard axis {shard_axis}")
        print("", f"MFU: {mfu:.2f}")
        print("", "Accuracy:", jnp.mean(jnp.abs(result - result_.reshape(result.shape))), jnp.mean(jnp.abs(result)))
