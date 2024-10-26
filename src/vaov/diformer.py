# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
import equinox as eqx
from equinox import nn
from dataclasses import dataclass
from jaxtyping import Array, Float, UInt
from typing import Optional
import jax.numpy as jnp
from einops import rearrange
import jax
import math

@dataclass
class DiFormerConfig:
    """Configuration class for the diffusion transformer."""
    in_channels: int = 64
    time_embed_dim: int = 256
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple[int] = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    guidance_embed: bool = True
    guidance_embed_dim: int = 256


def rope(pos: Float[Array, "*batch n_seq"], dim: int, theta: int) -> Float[Array, "*batch n_seq {dim} 2 2"]:
    """
    Compute rotary position embeddings.

    Args:
        pos: Position tensor.
        dim: Dimension of the embeddings.
        theta: Scaling factor for frequencies.

    Returns:
        Rotary position embeddings.
    """
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

    Args:
        t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim: The dimension of the output.
        max_period: Controls the minimum frequency of the embeddings.
        time_factor: Scaling factor for time.

    Returns:
        An (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(0, half, dtype=jnp.float32)
        / half
    )

    args = t[..., None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


class EmbedND(eqx.Module):
    """N-dimensional embedding module."""
    dim: int
    theta: int
    axes_dim: list[int]

    def __call__(self, ids: Float[Array, "*batch n_seq"]) -> Float[Array, "*batch n_seq n_heads embed_dim"]:
        """
        Compute N-dimensional embeddings.

        Args:
            ids: Input tensor.

        Returns:
            N-dimensional embeddings.
        """
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )
        return emb.unsqueeze(1)


class VLinear(nn.Linear):
    """Linear layer with optional bias."""
    in_channels: int
    out_channels: int
    weight: Float[Array, "in_channels out_channels"]
    bias: Optional[Float[Array, "out_channels"]]

    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = False, *, key: jax.random.PRNGKey):
        """
        Initialize VLinear layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_bias: Whether to use bias.
            key: Random key for initialization.
        """
        super().__init__(in_channels, out_channels, use_bias=False, key=key)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = jax.random.normal(key, (in_channels, out_channels)) * (1 / math.sqrt(in_channels))
        if use_bias:
            self.bias = jax.random.normal(key, (out_channels))
        else:
            self.bias = None

    def __call__(self, x: Float[Array, "*batch in_channels"]) -> Float[Array, "*batch out_channels"]:
        """
        Apply linear transformation.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class MLPEmbedder(eqx.Module):
    """MLP-based embedder."""
    in_layer: VLinear
    out_layer: VLinear
    
    def __init__(self, in_dim: int, hidden_dim: int, *, key: jax.random.PRNGKey):
        """
        Initialize MLPEmbedder.

        Args:
            in_dim: Input dimension.
            hidden_dim: Hidden dimension.
            key: Random key for initialization.
        """
        super().__init__()
        self.in_layer = VLinear(in_dim, hidden_dim, use_bias=True, key=key)
        self.out_layer = VLinear(hidden_dim, hidden_dim, use_bias=True, key=key)

    def __call__(
        self, x: Float[Array, "*batch vec_in_dim"]
    ) -> Float[Array, "*batch hidden_dim"]:
        """
        Apply MLP embedding.

        Args:
            x: Input tensor.

        Returns:
            Embedded tensor.
        """
        return self.out_layer(jax.nn.silu(self.in_layer(x)))


class DiFormer(eqx.Module):
    """Diffusion transformer."""
    config: DiFormerConfig
    pe_embedder: EmbedND
    time_in: MLPEmbedder
    img_in: VLinear
    txt_in: VLinear
    vector_in: MLPEmbedder
    guidance_in: eqx.Module

    def __init__(self, config: DiFormerConfig, key: jax.random.PRNGKey):
        """
        Initialize diffusion transformer.

        Args:
            config: Configuration for the model.
            key: Random key for initialization.
        """
        self.config = config
        
        pe_dim = config.hidden_size // config.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )
        self.time_in = MLPEmbedder(self.config.time_embed_dim, config.hidden_size, key=key)
        self.img_in = VLinear(
            config.in_channels, config.hidden_size, use_bias=True, key=key
        )
        self.txt_in = VLinear(config.context_in_dim, config.hidden_size, key=key)
        self.vector_in = MLPEmbedder(config.vec_in_dim, config.hidden_size, key=key)
        self.guidance_in = (
            MLPEmbedder(in_dim=self.config.guidance_embed_dim, hidden_dim=self.config.hidden_size, key=key) if config.guidance_embed else nn.Identity()
        )

    def __call__(
        self,
        img: Float[Array, "*batch n_seq in_channels"],
        txt: Float[Array, "*batch n_seq_txt context_in_dim"],
        timesteps: Float[Array, "*batch"],
        y: Float[Array, "*batch vec_in_dim"],
        img_ids: UInt[Array, "*batch n_seq 3"],
        txt_ids: Optional[UInt[Array, "*batch n_seq_txt 3"]] = None,
        guidance: Optional[Float[Array, "*batch"]] = None,
    ):
        """
        Forward pass of the diffusion transformer.

        Args:
            img: Image input tensor.
            txt: Text input tensor.
            timesteps: Timestep tensor.
            y: Vector input tensor.

        Returns:
            Processed image tensor.
        """
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, self.config.time_embed_dim))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, self.config.guidance_embed_dim))
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)
        
        if txt_ids is None:
            txt_ids = jnp.zeros(txt.shape[:-1] + (3,), dtype=jnp.uint32)
        ids = jnp.concatenate((txt_ids, img_ids), axis=-2)
        pe = self.pe_embedder(ids)
        
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        
        return img


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        model = DiFormer(DiFormerConfig(), jax.random.PRNGKey(0))
        
        n_batch = 1
        h, w = 32, 32
        n_seq = h * w
        n_seq_txt = 128
        
        key = jax.random.PRNGKey(0)

        img = jax.random.normal(key, (n_batch, n_seq, model.config.in_channels))
        txt = jax.random.normal(key, (n_batch, n_seq_txt, model.config.context_in_dim))
        timesteps = jax.random.uniform(key, (n_batch,))

        img_ids = jnp.zeros((n_batch, h, w, 3), dtype=jnp.uint32)
        img_ids = img_ids.at[..., 1].add(jnp.arange(h)[:, None])
        img_ids = img_ids.at[..., 2].add(jnp.arange(w)[None, :])
        img_ids = img_ids.reshape(n_batch, -1, 3)

        guidance_scale = jnp.full((n_batch,), 10)
        y = jax.random.normal(key, (n_batch, model.config.vec_in_dim))

        print(model(img=img, txt=txt, timesteps=timesteps, y=y, img_ids=img_ids, guidance=guidance_scale).shape)
