# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
import equinox as eqx
from dataclasses import dataclass
from jaxtyping import Array, Float
import jax.numpy as jnp
from einops import rearrange
import jax
import math

@dataclass
class DiFormerConfig:
    in_channels: int = 64
    time_embed_dim: int = 256
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list[int] = [16, 56, 56]
    theta: int = 10_000
    qkv_bias: bool = True
    guidance_embed: bool = True


def rope(pos: Float[Array, "*batch n_seq"], dim: int, theta: int) -> Float[Array, "*batch n_seq {dim} 2 2"]:
    assert dim % 2 == 0
    scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    omega = 1.0 / (theta**scale)
    out = jnp.einsum("...n,d->...nd", pos, omega)
    out = jnp.stack(
        [jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1
    )
    out = rearrange(out, "... n d (i j) -> ... n d i j", i=2, j=2)
    
    return out.float()


def timestep_embedding(
    t: Float[Array, "*batch"],
    dim: int, max_period=10000,
    time_factor: float = 1000.0) -> Float[Array, "*batch {dim}"]:
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(start=0, end=half, dtype=jnp.float32)
        / half
    )

    args = t[..., None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], dim=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


class EmbedND(eqx.Module):
    dim: int
    theta: int
    axes_dim: list[int]

    def __call__(self, ids: Float[Array, "*batch n_seq"]) -> Float[Array, "*batch n_seq n_heads embed_dim"]:
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )
        return emb.unsqueeze(1)


class MLPEmbedder(eqx.Module):
    in_layer: eqx.nn.Linear
    out_layer: eqx.nn.Linear
    
    def __init__(self, in_dim: int, hidden_dim: int, key: jax.random.PRNGKey):
        super().__init__()
        self.in_layer = eqx.nn.Linear(in_dim, hidden_dim, use_bias=True, key=key)
        self.out_layer = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=True, key=key)

    def forward(
        self, x: Float[Array, "*batch vec_in_dim"]
    ) -> Float[Array, "*batch hidden_dim"]:
        return self.out_layer(jax.nn.silu(self.in_layer(x)))


class DiFormer(eqx.Module):
    config: DiFormerConfig

    def __init__(self, config: DiFormerConfig, key: jax.random.PRNGKey):
        self.config = config
        
        pe_dim = config.hidden_size // config.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )
        self.img_in = eqx.Linear(self.in_channels, self.hidden_size, use_bias=True, key=key)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size, key=key)

    def __call__(
        self,
        img: Float[Array, "*batch n_seq in_channels"],
        timesteps: Float[Array, "*batch"],
        y: Float[Array, "*batch vec_in_dim"],
    ):
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, self.config.time_embed_dim))
        vec = vec + self.vector_in(y)

        
