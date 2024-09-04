import jax
import jax.numpy as jnp
import jax.numpy as np
import jax
from math import ceil
from functools import partial
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from matplotlib import pyplot as plt
import timeit


v4_flops = 275e12
def benchmark(f, ntrials: int = 10):
    def run(*args, **kwargs):
        # Compile function first
        jax.block_until_ready(f(*args, **kwargs))
        # Time function
        result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                                number=ntrials)
        time = result / ntrials
        # print(f"Time: {time}")
        return time
    return run

def test_min_dim():
    def simple_matmul(a_ref, b_ref, out_ref):
        out_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
    N = 16
    K = 16
    M = 16
    A = jax.random.normal(jax.random.PRNGKey(0), (N, K), dtype=jnp.bfloat16)
    B = jax.random.normal(jax.random.PRNGKey(1), (K, M), dtype=jnp.bfloat16).astype(jnp.int4)
    C = pl.pallas_call(
        simple_matmul,
        grid=(1, 1, 1),
        out_shape=jax.ShapeDtypeStruct(
            (N, M), jnp.float32
        ),
        interpret=False
    )(A, B).block_until_ready()

def matmul_flops(m: int, k: int, n: int):
    return 2 * m * k * n


def int4_matmul_mfu():
    def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        # dot_general expects a data structure (contraction_dims, batch_dims),
        # where contraction_dims are the set of dimensions for LHS and RHS that will
        # be contracted (reduced) in the matmul; batch_dims, on the other hand, are
        # looped over. The remaining dimensions will be the input and output dimension
        # of the matmul.
        if transpose_rhs:
            dims = ((1,), (1,)), ((), ())
        else:
            dims = ((1,), (0,)), ((), ())

        acc_ref[...] += jax.lax.dot_general(
            x_ref[...],
            y_ref[...],
            dims,
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(2) == nsteps - 1)
        def _():
            z_ref[...] = acc_ref[...].astype(z_ref.dtype)


    @partial(jax.jit, static_argnames=["bm", "bk", "bn", "transpose_rhs"])
    def matmul(
        x: jax.Array,
        y: jax.Array,
        *,
        bm: int = 128,
        bk: int = 128,
        bn: int = 128,
        transpose_rhs: bool = False,
    ):
        if transpose_rhs:
            y = y.swapaxes(0, 1)
            y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
        else:
            y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
        m, k = x.shape
        _, n = y.shape
        return pl.pallas_call(
            partial(matmul_kernel, nsteps=k // bk, transpose_rhs=transpose_rhs),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                    y_block_spec,
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
                scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
                grid=(m // bm, n // bn, k // bk),
            ),
            out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
            compiler_params=dict(
                mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
            ),
        )(x, y)

    def analyze_matmul(
        bm: int, bk: int, bn: int, dtype: np.dtype, rside_dtype: np.dtype, transpose_rhs: bool = False
    ):
        m, k, n = 8192, 8192, 8192
        mm_func = partial(matmul, bm=bm, bk=bk, bn=bn)
        x = jnp.ones((m, k), dtype=dtype)
        if transpose_rhs:
            y = jnp.ones((n, k), dtype=rside_dtype)

            @jax.jit
            def _wrapper(x, y):
                y = y.swapaxes(0, 1)
                return mm_func(x, y, transpose_rhs=True)
        else:
            y = jnp.ones((k, n), dtype=rside_dtype)
            _wrapper = mm_func
        time = benchmark(_wrapper)(x, y)
        print(f"----- {bm} x {bk} x {bn} ({dtype} x {rside_dtype}) -----")
        print("Matmul time: ", time)
        mm_flops = matmul_flops(m, k, n) / time
        print("Matmul FLOP/s: ", mm_flops)
        print(f"FLOP/s utilization: {mm_flops / v4_flops * 100:.4f}%")
        print()

    return analyze_matmul

def quant_2d():
    x = jax.random.normal(jax.random.PRNGKey(0), (128, 128), dtype=jnp.float32)
    d_1 = jnp.ones(x.shape[0], dtype=jnp.float32)
    d_2 = jnp.ones(x.shape[1], dtype=jnp.float32)
    # https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf
    # 2017 Daniel Ruiz "A scaling algorithm to equilibrate both rows and columns norms in matrices"
    # scaling, huh?
    def update(_, state):
        x, d_1, d_2 = state
        d_r = jnp.sqrt(jnp.abs(x).max(axis=1))
        d_c = jnp.sqrt(jnp.abs(x).max(axis=0))
        x = x / d_r[:, None] / d_c[None, :]
        d_1 = d_1 / d_r
        d_2 = d_2 / d_c
        return x, d_1, d_2
    x, d_1, d_2 = jax.lax.fori_loop(0, 100, update, (x, d_1, d_2))
    plt.subplot(221)
    plt.imshow(x)
    plt.colorbar()
    plt.subplot(222)
    plt.hist(d_1)
    plt.subplot(223)
    plt.hist(d_2)
    plt.subplot(224)
    plt.hist(x.ravel(), bins=100)
    plt.savefig("misc/2quant.png")

if __name__ == "__main__":
    pass
