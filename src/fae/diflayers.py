from dataclasses import dataclass
import math
import threading
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from oryx.core import sow
from einops import rearrange
from jax.experimental.pallas.ops.tpu import flash_attention
import jax.experimental
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float
import equinox as eqx
from equinox import nn
from functools import partial
from .interp_globals import post_double_stream


@dataclass(frozen=True)
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

    @property
    def mlp_size(self):
        return int(self.hidden_size * self.mlp_ratio)


def rope(
    pos: Float[Array, "*batch n_seq"], dim: int, theta: int
) -> Float[Array, "*batch n_seq {dim} 2 2"]:
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
    out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1)
    out = rearrange(out, "... n d (i j) -> ... n d i j", i=2, j=2)
    return out.astype(jnp.float32)


global_mesh = threading.local()


def attention(
    q: Float[Array, "*batch n_seq n_heads d_head"],
    k: Float[Array, "*batch n_seq n_heads d_head"],
    v: Float[Array, "*batch n_seq n_heads d_head"],
    pe: Float[Array, "*batch n_seq d_head 2 2"],
    mask: Optional[Bool[Array, "*batch n_seq n_seq"]] = None,
) -> Float[Array, "*batch n_seq (n_heads d_head)"]:
    if q.ndim > 4:
        if mask is not None:  # flash
            assert q.ndim == 5

            def per_block(q, k, v, pe, mask):
                og_batch_dims = q.shape[:-3]
                q = q.reshape(-1, *q.shape[-3:])
                k = k.reshape(-1, *k.shape[-3:])
                v = v.reshape(-1, *v.shape[-3:])
                pe = pe.reshape(-1, *pe.shape[len(og_batch_dims) :])
                mask = mask.reshape(-1, *mask.shape[len(og_batch_dims) :])
                x = attention(q, k, v, pe, mask)
                return x.reshape(*og_batch_dims, *x.shape[-2:])

            args = (q, k, v, pe, mask)
            return jax.experimental.shard_map.shard_map(
                per_block,
                global_mesh.mesh,
                in_specs=tuple(
                    P("dp", "fsdp", *((None,) * (x.ndim - 3))) for x in args
                ),
                out_specs=P("dp", "fsdp", None, None),
                check_rep=False,
            )(*args)
        else:
            return jax.vmap(attention, in_axes=(0, 0, 0, 0))(q, k, v, pe)

    q, k = apply_rope(q, k, pe)
    if jax.devices()[0].platform == "cpu":
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        return rearrange(jax.nn.dot_product_attention(q, k, v), "... h n -> ... (h n)")

    q = q / math.sqrt(q.shape[-1])

    segment_ids = None
    if mask is not None:
        q_segment_ids = mask.astype(jnp.int32)
        kv_segment_ids = mask.astype(jnp.int32)
        segment_ids = flash_attention.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
    x = flash_attention.flash_attention(q, k, v, segment_ids=segment_ids)
    x = rearrange(x, "... h s d -> ... s (h d)")

    return x


def apply_rope(
    xq: Float[Array, "*batch (d_model 2)"],
    xk: Float[Array, "*batch (d_model 2)"],
    freqs_cis: Float[Array, "*batch d_model 2 2"],
) -> tuple[Float[Array, "*batch d_model"], Float[Array, "*batch d_model"]]:
    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).astype(xq), xk_out.reshape(*xk.shape).astype(xk)


def timestep_embedding(
    t: Float[Array, "*batch"], dim: int, max_period=10000, time_factor: float = 1000.0
) -> Float[Array, "*batch {dim}"]:
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
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
    )

    args = t[..., None].astype(jnp.float32) * freqs
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[..., :1])], axis=-1
        )
    return embedding


class EmbedND(eqx.Module):
    """N-dimensional embedding module."""

    dim: int
    theta: int
    axes_dim: list[int]

    def __call__(
        self, ids: Float[Array, "*batch n_seq n_axes"]
    ) -> Float[Array, "*batch n_heads n_seq embed_dim 2 2"]:
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
            # "*batch n_seq {dim} 2 2"
            axis=-3,
        )
        return emb[..., None, :, :, :, :]


class VLinear(eqx.Module):
    """Linear layer with optional bias."""

    in_channels: int
    out_channels: int
    weight: Float[Array, "in_channels out_channels"]
    bias: Optional[Float[Array, "out_channels"]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = False,
        *,
        key: jax.random.PRNGKey,
    ):
        """
        Initialize VLinear layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_bias: Whether to use bias.
            key: Random key for initialization.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = jax.random.normal(key, (in_channels, out_channels)) * (
            1 / math.sqrt(in_channels)
        )
        if use_bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(
        self, x: Float[Array, "*batch in_channels"], **kwargs
    ) -> Float[Array, "*batch out_channels"]:
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


class VLayerNorm(nn.LayerNorm):
    def __call__(self, x, **kwargs):
        if len(x.shape) > len(self.shape):
            return jax.vmap(self)(x, **kwargs)
        return super().__call__(x, **kwargs)


class LastLayer(eqx.Module):
    norm_final: VLayerNorm
    linear: VLinear
    adaLN_modulation: eqx.Module

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.norm_final = VLayerNorm(
            hidden_size, use_weight=False, use_bias=False, eps=1e-6
        )
        self.linear = VLinear(
            hidden_size, patch_size * patch_size * out_channels, use_bias=True, key=key
        )
        self.adaLN_modulation = nn.Sequential(
            (
                nn.Lambda(jax.nn.silu),
                VLinear(hidden_size, 2 * hidden_size, use_bias=True, key=key),
            )
        )

    def __call__(
        self, x: Float[Array, "*batch n_seq hidden_size"], vec: "*batch hidden_size"
    ) -> Float[Array, "*batch n_seq hidden_size"]:
        shift, scale = jnp.split(self.adaLN_modulation(vec), 2, axis=-1)
        x = (1 + scale[..., None, :]) * self.norm_final(x) + shift[..., None, :]
        x = self.linear(x)
        return x


class QKNorm(eqx.Module):
    query_norm: nn.RMSNorm
    key_norm: nn.RMSNorm

    def __init__(self, dim: int):
        # what's the point of weights on both queries and keys?..
        self.query_norm = nn.RMSNorm(dim, use_weight=True, use_bias=False)
        self.key_norm = nn.RMSNorm(dim, use_weight=True, use_bias=False)

    def apply_norm(
        self, x: Float[Array, "*batch dim"], norm: nn.RMSNorm
    ) -> Float[Array, "*batch dim"]:
        if len(x.shape) > len(norm.shape):
            return jax.vmap(self.apply_norm, in_axes=(0, None))(x, norm)
        return norm(x)

    def __call__(
        self,
        q: Float[Array, "*batch dim"],
        k: Float[Array, "*batch dim"],
        v: Float[Array, "*batch dim"],
    ) -> Tuple[Float[Array, "*batch dim"], Float[Array, "*batch dim"]]:
        q = self.apply_norm(q, self.query_norm)
        k = self.apply_norm(k, self.key_norm)
        return q.astype(v), k.astype(v)


@dataclass
class ModulationOut:
    shift: Float[Array, "*batch seq dim"]
    scale: Float[Array, "*batch seq dim"]
    gate: Float[Array, "*batch seq dim"]

    def __call__(
        self, x: Float[Array, "*batch seq dim"]
    ) -> Float[Array, "*batch seq dim"]:
        return (1 + self.scale) * x + self.shift


class Modulation(eqx.Module):
    is_double: bool
    multiplier: int
    lin: VLinear

    def __init__(self, dim: int, double: bool, *, key: jax.random.PRNGKey):
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = VLinear(dim, self.multiplier * dim, use_bias=True, key=key)

    def __call__(
        self, vec: Float[Array, "*batch dim"]
    ) -> tuple[ModulationOut, ModulationOut | None]:
        lin_of_silu = self.lin(jax.nn.silu(vec))
        out = jnp.split(lin_of_silu[..., None, :], self.multiplier, axis=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class SelfAttention(eqx.Module):
    """Self-attention"""

    qk_norm: QKNorm
    qkv_proj: VLinear
    o_proj: VLinear
    head_dim: int

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        *,
        key: jax.random.PRNGKey,
    ):
        self.head_dim = dim // num_heads
        self.qkv_proj = VLinear(dim, dim * 3, use_bias=qkv_bias, key=key)
        self.o_proj = VLinear(dim, dim, use_bias=True, key=key)
        self.qk_norm = QKNorm(self.head_dim)

    def qkv(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv,
            "... seq (qkv heads dim) -> qkv ... heads seq dim",
            qkv=3,
            dim=self.head_dim,
        )
        q, k = self.qk_norm(q, k, v)
        return q, k, v

    def __call__(self, x, pe, mask=None, no_out=False):
        q, k, v = self.qkv(x)
        attn = attention(q, k, v, pe=pe, mask=mask)
        if no_out:
            return attn
        return self.o_proj(attn)


class MLP(eqx.Module):
    in_proj: VLinear
    out_proj: VLinear

    def __init__(self, config: DiFormerConfig, *, key: jax.random.PRNGKey):
        mlp_hidden_dim = config.mlp_size
        self.in_proj = VLinear(
            config.hidden_size,
            mlp_hidden_dim,
            key=key,
            use_bias=True,
        )
        self.out_proj = VLinear(
            mlp_hidden_dim, config.hidden_size, key=key, use_bias=True
        )

    def __call__(self, x, no_out=False):
        mlp = self.in_proj(x)
        mlp = jax.nn.gelu(mlp, approximate=True)
        if no_out:
            return mlp
        return self.out_proj(mlp)


def fg(x):
    """Fix gated addition."""
    # return jnp.clip(x, -32768, 32768)
    return x


def fr(x):
    """Fix residual."""
    # return jnp.clip(x, -32768, 32768)
    return x


class DoubleStreamBlock(eqx.Module):
    """Main two-stream MMDiT block."""

    config: DiFormerConfig
    img_mod: Modulation
    img_norm1: VLayerNorm
    img_attn: SelfAttention
    img_norm2: VLayerNorm
    img_mlp: MLP
    txt_mod: Modulation
    txt_norm1: VLayerNorm
    txt_attn: SelfAttention
    txt_norm2: VLayerNorm
    txt_mlp: MLP

    def __init__(self, config: DiFormerConfig, *, key: jax.random.PRNGKey):
        self.config = config
        self.img_norm1, self.img_norm2, self.txt_norm1, self.txt_norm2 = (
            VLayerNorm(config.hidden_size, use_weight=False, use_bias=False, eps=1e-6)
            for _ in range(4)
        )
        self.img_mod, self.txt_mod = (
            Modulation(config.hidden_size, double=True, key=k)
            for k in jax.random.split(key, 2)
        )
        self.img_attn, self.txt_attn = (
            SelfAttention(
                config.hidden_size, config.num_heads, qkv_bias=config.qkv_bias, key=k
            )
            for k in jax.random.split(key, 2)
        )
        self.img_mlp, self.txt_mlp = (
            MLP(config, key=k) for k in jax.random.split(key, 2)
        )

    def __call__(self, data, vec, pe, mask=None, layer_idx=None):
        img, txt = data["img"], data["txt"]
        txt_len = txt.shape[-2]

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_normed = self.img_norm1(img)
        img_modulated = img_mod1(img_normed)
        img_q, img_k, img_v = self.img_attn.qkv(img_modulated)

        txt_normed = self.txt_norm1(txt)
        txt_modulated = txt_mod1(txt_normed)
        txt_q, txt_k, txt_v = self.txt_attn.qkv(txt_modulated)

        q = jnp.concatenate((txt_q, img_q), axis=-2)
        k = jnp.concatenate((txt_k, img_k), axis=-2)
        v = jnp.concatenate((txt_v, img_v), axis=-2)

        attn = attention(q, k, v, pe, mask=mask)
        txt_attn, img_attn = attn[..., :txt_len, :], attn[..., txt_len:, :]

        img_out1 = img_mod1.gate * self.img_attn.o_proj(img_attn)
        img = img + fg(img_out1)
        img_out2 = img_mod2.gate * self.img_mlp(img_mod2(self.img_norm2(img)))
        img = img + fg(img_out2)

        txt_out1 = txt_mod1.gate * self.txt_attn.o_proj(txt_attn)
        txt = txt + fg(txt_out1)
        txt_out2 = txt_mod2.gate * self.txt_mlp(txt_mod2(self.txt_norm2(txt)))
        txt = txt + fg(txt_out2)

        sow(img, tag="interp", name="double_img", mode="append")
        sow(txt, tag="interp", name="double_txt", mode="append")

        result = dict(img=fr(img), txt=fr(txt))
        post_double_stream.jax_callback(layer_idx, result)
        return result


class SingleStreamBlock(eqx.Module):
    """Main single-stream MMDiT block."""

    config: DiFormerConfig
    head_dim: int

    attn: SelfAttention
    mlp: MLP

    pre_norm: VLayerNorm
    modulation: Modulation

    def __init__(self, config: DiFormerConfig, *, key: jax.random.PRNGKey):
        self.config = config

        self.head_dim = config.hidden_size // config.num_heads

        self.attn = SelfAttention(
            config.hidden_size, config.num_heads, qkv_bias=config.qkv_bias, key=key
        )
        self.mlp = MLP(config, key=key)

        self.pre_norm = VLayerNorm(
            config.hidden_size, use_weight=False, use_bias=False, eps=1e-6
        )

        self.modulation = Modulation(config.hidden_size, double=False, key=key)

    def __call__(self, data, vec, pe, mask=None, layer_idx=None):
        mod, _ = self.modulation(vec)
        x = self.pre_norm(data)
        x = mod(x)

        attn_out = self.attn(x, pe, mask=mask)

        mlp_out = self.mlp(x)

        par_out = attn_out + mlp_out
        out = data + fg(mod.gate * par_out)
        out = fr(out)

        return out
