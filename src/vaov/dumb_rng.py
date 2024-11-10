from functools import partial
import jax
import jax.numpy as jnp
from jax._src import prng


@jax.jit
def dumb_seed(seed):
    # sorry for being wasteful, it has to be int32 and non-empty
    return jnp.empty((1), dtype=jnp.uint32)


# https://github.com/jax-ml/jax/blob/12d26053e31c5c9f45da5a15ce7fb7fcbb0a96b7/jax/_src/prng.py#L1098
@partial(jax.jit, static_argnums=(1,))
def dumb_split(key, shape):
    return jnp.zeros(shape + (1,), dtype=jnp.uint32)


@jax.jit
def dumb_fold_in(key, data):
    return key


# https://github.com/jax-ml/jax/blob/12d26053e31c5c9f45da5a15ce7fb7fcbb0a96b7/jax/_src/prng.py#L65
UINT_DTYPES = {8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64}


# https://github.com/jax-ml/jax/blob/12d26053e31c5c9f45da5a15ce7fb7fcbb0a96b7/jax/_src/prng.py#L1151
@partial(jax.jit, static_argnums=(1, 2))
def dumb_bits(key, bit_width, shape):
    return jnp.zeros(shape, dtype=UINT_DTYPES[bit_width])


dumb_prng_impl = prng.PRNGImpl(
    key_shape=(1,),
    seed=dumb_seed,
    split=dumb_split,
    random_bits=dumb_bits,
    fold_in=dumb_fold_in,
)
