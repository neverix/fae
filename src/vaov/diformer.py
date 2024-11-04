# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
import equinox as eqx
from equinox import nn
import jax.experimental
import jax.experimental.shard_map
from safetensors.numpy import load_file
import ml_dtypes
import shutil
from functools import partial
from dataclasses import dataclass
from collections import defaultdict
from jaxtyping import Array, Float, UInt, Bool
from typing import Optional, Tuple
from .quant import QuantMatrix
import qax.primitives
import qax
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from jax.experimental.pallas.ops.tpu import flash_attention
from jax._src import prng
import jax.numpy as jnp
from tqdm.auto import tqdm
from .vae import FluxVAE
from einops import rearrange
import jax
import orbax.checkpoint as ocp
from pathlib import Path
import math
import numpy as np
from loguru import logger
from typing import TypeVar, Generic

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
    return out.astype(jnp.float32)


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
                pe = pe.reshape(-1, *pe.shape[len(og_batch_dims):])
                mask = mask.reshape(-1, *mask.shape[len(og_batch_dims):])
                x = attention(q, k, v, pe, mask)
                # print(q.shape, k.shape, v.shape, pe.shape, mask.shape, x.shape)
                return x.reshape(*og_batch_dims, *x.shape[-2:])
            args = (q, k, v, pe, mask)
            return jax.experimental.shard_map.shard_map(
                per_block, mesh,
                in_specs=tuple(P("dp", "fsdp", *((None,) * (x.ndim - 3))) for x in args),
                out_specs=P("dp", "fsdp", None, None),
                check_rep=False
            )(*args)
        else:
            return jax.vmap(attention, in_axes=(0, 0, 0, 0))(q, k, v, pe)

    q, k = apply_rope(q, k, pe)

    # x = jax.nn.dot_product_attention(q, k, v, mask=mask)
    # x = rearrange(x, "... h d -> ... (h d)")
    segment_ids = None
    if mask is not None:
        q_segment_ids = mask.astype(jnp.int32)
        kv_segment_ids = mask.astype(jnp.int32)
        segment_ids = flash_attention.SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
    q = q.transpose((0, 2, 1, 3))
    k = k.transpose((0, 2, 1, 3))
    v = v.transpose((0, 2, 1, 3))
    x = flash_attention.flash_attention(q, k, v, segment_ids=segment_ids)
    x = rearrange(x, "... h s d -> ... s (h d)")

    return x


def apply_rope(xq: Float[Array, "*batch (d_model 2)"], xk: Float[Array, "*batch (d_model 2)"],
               freqs_cis: Float[Array, "*batch d_model 2 2"]) -> tuple[
                   Float[Array, "*batch d_model"], Float[Array, "*batch d_model"]]:
    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).astype(xq), xk_out.reshape(*xk.shape).astype(xk)


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
            # "*batch n_seq {dim} 2 2"
            axis=-3,
        )
        return emb[..., None, :, :, :]


class VLinear(eqx.Module):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = jax.random.normal(key, (in_channels, out_channels)) * (1 / math.sqrt(in_channels))
        if use_bias:
            self.bias = jnp.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: Float[Array, "*batch in_channels"], **kwargs) -> Float[Array, "*batch out_channels"]:
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

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, *, key: jax.random.PRNGKey):
        self.norm_final = VLayerNorm(hidden_size, use_weight=False, use_bias=False, eps=1e-6)
        self.linear = VLinear(hidden_size, patch_size * patch_size * out_channels, use_bias=True, key=key)
        self.adaLN_modulation = nn.Sequential((
            nn.Lambda(jax.nn.silu),
            VLinear(hidden_size, 2 * hidden_size, use_bias=True, key=key)
        ))

    def __call__(self, x: Float[Array, "*batch n_seq hidden_size"], vec: "*batch hidden_size") -> Float[Array, "*batch n_seq hidden_size"]:
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

    def apply_norm(self, x: Float[Array, "*batch dim"], norm: nn.RMSNorm) -> Float[Array, "*batch dim"]:
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
    
    def __call__(self, x: Float[Array, "*batch seq dim"]) -> Float[Array, "*batch seq dim"]:
        return (1 + self.scale) * x + self.shift


class Modulation(eqx.Module):
    is_double: bool
    multiplier: int
    lin: VLinear

    def __init__(self, dim: int, double: bool, *, key: jax.random.PRNGKey):
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = VLinear(dim, self.multiplier * dim, use_bias=True, key=key)

    def __call__(self, vec: Float[Array, "*batch dim"]) -> tuple[ModulationOut, ModulationOut | None]:
        out = jnp.split(self.lin(jax.nn.silu(vec))[..., None, :], self.multiplier, axis=-1)

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
    
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, *, key: jax.random.PRNGKey):
        self.head_dim = dim // num_heads
        self.qkv_proj = VLinear(dim, dim * 3, use_bias=qkv_bias, key=key)
        self.o_proj = VLinear(dim, dim, use_bias=True, key=key)
        self.qk_norm = QKNorm(self.head_dim)

    def qkv(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv,
            "... (qkv heads dim) -> qkv ... heads dim",
            qkv=3,
            dim=self.head_dim,
        )
        q, k = self.qk_norm(q, k, v)
        return q, k, v

    def __call__(self, x, pe, mask=None):
        q, k, v = self.qkv(x)
        attn = attention(q, k, v, pe=pe, mask=mask)
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
            mlp_hidden_dim,
            config.hidden_size,
            key=key,
            use_bias=True
        )
    
    def __call__(self, x):
        mlp = self.in_proj(x)
        mlp = jax.nn.gelu(mlp, approximate=True)
        return self.out_proj(mlp)


class DoubleStreamBlock(eqx.Module):
    """Main two-stream MMDiT block."""

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
        self.img_norm1, self.img_norm2, self.txt_norm1, self.txt_norm2 = \
            (VLayerNorm(config.hidden_size, use_weight=False, use_bias=False, eps=1e-6) for _ in range(4))
        self.img_mod, self.txt_mod = (
            Modulation(config.hidden_size, double=True, key=k) for k in jax.random.split(key, 2))
        self.img_attn, self.txt_attn = (
            SelfAttention(config.hidden_size, config.num_heads, qkv_bias=config.qkv_bias, key=k) for k in jax.random.split(key, 2))
        self.img_mlp, self.txt_mlp = (
            MLP(config, key=k) for k in jax.random.split(key, 2)
        )

    def __call__(self, data, vec, pe, mask=None):
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
        
        q = jnp.concatenate((txt_q, img_q), axis=-3)
        k = jnp.concatenate((txt_k, img_k), axis=-3)
        v = jnp.concatenate((txt_v, img_v), axis=-3)
        
        attn = attention(q, k, v, pe, mask=mask)
        txt_attn, img_attn = attn[..., :txt_len, :], attn[..., txt_len:, :]

        img = img + img_mod1.gate * self.img_attn.o_proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(img_mod2(self.img_norm2(img)))
    
        txt = txt + txt_mod1.gate * self.txt_attn.o_proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(txt_mod2(self.txt_norm2(txt)))
        
        return dict(img=img, txt=txt)


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
        
        self.attn = SelfAttention(config.hidden_size, config.num_heads, qkv_bias=config.qkv_bias, key=key)
        self.mlp = MLP(config, key=key)

        self.pre_norm = VLayerNorm(config.hidden_size, use_weight=False, use_bias=False, eps=1e-6)
        
        self.modulation = Modulation(config.hidden_size, double=False, key=key)
    
    def __call__(self, data, vec, pe, mask=None):
        mod, _ = self.modulation(vec)
        x = self.pre_norm(data)
        x = mod(x)
        
        attn_out = self.attn(x, pe, mask=mask)
        mlp_out = self.mlp(x)

        return x + mod.gate * (attn_out + mlp_out)


def is_arr(x):
    return isinstance(x, qax.primitives.ArrayValue)


def unify(arg, *args, repeat=None):
    if not is_arr(arg):
        return arg
    if isinstance(arg, QuantMatrix):
        if repeat is not None:
            args = (arg,) * (repeat - 1)
        return arg.stack(*args)
    if repeat is not None:
        return jnp.repeat(arg[None], repeat, axis=0)
    return jnp.stack((arg, *args), axis=0)


T = TypeVar('T', bound=eqx.Module)

class SequentialScan(eqx.Module, Generic[T]):
    logic: T
    weights: T

    def __init__(self, layers: tuple[T, ...], repeat: int = None):
        unified = jax.tree.map(partial(unify, repeat=repeat), *layers, is_leaf=is_arr)
        self.weights, self.logic = eqx.partition(unified, is_arr)

    def __call__(self, x, *args, **kwargs):
        def scan_fn(carry, weight: T):
            layer = eqx.combine(weight, self.logic)
            return layer(carry, *args, **kwargs), None

        return jax.lax.scan(scan_fn, x, self.weights)[0]

    def __getattr__(self, name):
        return getattr(self.weights, name)


def pad_and_mask(x, pad_to=128):
    current_len = x.shape[-2]
    if current_len % pad_to == 0:
        return x, jnp.ones(x.shape[:-1], dtype=jnp.bool_)
    pad = pad_to - current_len % pad_to
    x = jnp.pad(x, ((0, 0),) * (x.ndim - 2) + ((0, pad), (0, 0)))
    mask = jnp.ones(x.shape[:-1], dtype=jnp.bool_)
    mask = mask.at[..., -pad].set(False)
    return x, mask


class DiFormer(eqx.Module):
    """Diffusion transformer."""
    config_cls = DiFormerConfig
    
    config: DiFormerConfig
    pe_embedder: EmbedND
    time_in: MLPEmbedder
    img_in: VLinear
    txt_in: VLinear
    vector_in: MLPEmbedder
    guidance_in: eqx.Module

    double_blocks: SequentialScan[DoubleStreamBlock]
    single_blocks: SequentialScan[SingleStreamBlock]
    final_layer: LastLayer

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
        self.txt_in = VLinear(
            config.context_in_dim, config.hidden_size, use_bias=True, key=key
        )
        self.vector_in = MLPEmbedder(config.vec_in_dim, config.hidden_size, key=key)
        self.guidance_in = (
            MLPEmbedder(in_dim=self.config.guidance_embed_dim, hidden_dim=config.hidden_size, key=key) if config.guidance_embed else nn.Identity()
        )

        # double_blocks = tuple(DoubleStreamBlock(config, key=k) for k in jax.random.split(key, config.depth))
        # self.double_blocks = SequentialScan(double_blocks)
        # single_blocks = tuple(SingleStreamBlock(config, key=k) for k in jax.random.split(key, config.depth_single_blocks))
        # self.single_blocks = SequentialScan(single_blocks)

        self.double_blocks = SequentialScan((DoubleStreamBlock(config, key=key),), repeat=config.depth)
        self.single_blocks = SequentialScan((SingleStreamBlock(config, key=key),), repeat=config.depth_single_blocks)
    
        self.final_layer = LastLayer(config.hidden_size, 1, config.in_channels, key=key)

    def __call__(
        self,
        img: Float[Array, "*batch n_seq in_channels"],
        txt: Float[Array, "*batch n_seq_txt context_in_dim"],
        timesteps: Float[Array, "*batch"],
        y: Float[Array, "*batch vec_in_dim"],
        img_ids: UInt[Array, "*batch n_seq 3"],
        txt_ids: Optional[UInt[Array, "*batch n_seq_txt 3"]] = None,
        guidance: Optional[Float[Array, "*batch"]] = None,
    ) -> Float[Array, "*batch n_seq (patch_size patch_size in_channels)"]:
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
        
        orig_img_len = img.shape[-2]
        img, img_mask = pad_and_mask(img)
        txt, txt_mask = pad_and_mask(txt)
        
        if txt_ids is None:
            txt_ids = jnp.zeros(txt.shape[:-1] + (3,), dtype=jnp.uint32)
        img_ids = pad_and_mask(img_ids)[0]
        ids = jnp.concatenate((txt_ids, img_ids), axis=-2)
        pe = self.pe_embedder(ids)
        
        mask = jnp.concatenate((txt_mask, img_mask), -1)
        data = dict(img=img, txt=txt)
        data = self.double_blocks(data, vec=vec, pe=pe, mask=mask)

        txt, img = data["txt"], data["img"]
        data = jnp.concatenate((txt, img), -2)
        data = self.single_blocks(data, vec=vec, pe=pe, mask=mask)
        img = img[..., :orig_img_len, :]

        img = self.final_layer(img, vec)
        
        return img

    @classmethod
    def from_pretrained(cls, *args, custom_config=None, **kwargs):
        logger.info("Loading DiFormer model")
        if custom_config is None:
            custom_config = cls.config_cls()
        model = cls(custom_config, jax.random.key(0, impl=dumb_prng_impl))
        return load_flux(model, *args, **kwargs)


def selector_fn(name):
    if "." in name:
        name, _, children = name.partition(".")
        child_selector = selector_fn(children)
    else:
        child_selector = lambda x: x
    def selector(obj):
        if all(c.isnumeric() for c in name):
            return child_selector(obj[int(name)])
        return child_selector(getattr(obj, name))
    return selector


def weight_slice(arr, *, axis: int, start: int, size: int):
    if isinstance(arr, QuantMatrix):
        return arr.slice(axis=axis, start=start, size=size)
    return jax.lax.dynamic_slice_in_dim(arr, start, size, axis=axis)


def preprocess_weights(flux, model):
    flux = {k.replace("model.diffusion_model.", ""): v for k, v in flux.items()}
    
    logger.info("Layer-stacking arrays")
    flux_aggs = defaultdict(dict)
    new_flux = {}
    for key, value in flux.items():
        if key.startswith("double_blocks") or key.startswith("single_blocks"):
            base, _, key = key.partition(".")
            index, _, key = key.partition(".")
            flux_aggs[f"{base}.{key}"][int(index)] = value
        else:
            new_flux[key] = value
    for key, values in flux_aggs.items():
        new_weight = jnp.stack(
            [values[i] for i in list(range(max(values.keys()) + 1))], 0
        )
        new_flux[key] = new_weight
    flux = new_flux

    flux = {k.replace("attn.norm", "attn.qk_norm"): v for k, v in flux.items()}
    flux = {k.replace("attn.proj", "attn.o_proj"): v for k, v in flux.items()}
    flux = {k.replace("attn.qkv.", "attn.qkv_proj."): v for k, v in flux.items()}
    flux = {k.replace("mlp.0.", "mlp.in_proj."): v for k, v in flux.items()}
    flux = {k.replace("mlp.2.", "mlp.out_proj."): v for k, v in flux.items()}

    # fold in nf4 arrays
    logger.info("Loading in quantized arrays")
    array_flux = {}
    nf4_flux = defaultdict(dict)
    for key, value in flux.items():
        if ".weight" in key:
            key, dot_weight, name = key.partition(".weight")
            nf4_flux[key + dot_weight][name] = value
        else:
            array_flux[key] = jnp.asarray(value)
    for key, values in nf4_flux.items():
        if set(values.keys()) == {""}:
            array_flux[key] = values[""]
            continue
        if key == "single_blocks.linear1.weight":
            og_shape = (
                model.config.depth_single_blocks,
                model.config.hidden_size,
                model.config.hidden_size * 3 + model.config.mlp_size,
            )
            og_dtype = jnp.bfloat16
        elif key == "single_blocks.linear2.weight":
            og_shape = (
                model.config.depth_single_blocks,
                model.config.hidden_size + model.config.mlp_size,
                model.config.hidden_size,
            )
            og_dtype = jnp.bfloat16
        else:
            try:
                og_tensor = selector_fn(key)(model)
            except (AttributeError, TypeError) as e:
                raise AttributeError(f"Can't get {key}") from e
            if og_tensor is None:
                raise AttributeError(f"{key} is not initialized")
            og_shape = og_tensor.shape
            og_dtype = og_tensor.dtype
        quants = values[""]
        assert quants.dtype == jnp.uint8
        og_size = quants.size
        # quants = jax.lax.bitcast_convert_type(quants, jnp.int4)
        high, low = jnp.divmod(quants, 16)
        quants = jnp.stack((high, low), -1).reshape(*quants.shape[:-1], -1)
        assert quants.size == og_size * 2
        scales = values[".absmax"]
        block_size = quants.size // scales.size
        # quants = quants.reshape(*quants.shape[:-2], -1, block_size, og_shape[-1])
        # scales = scales.reshape(*scales.shape[:-1], -1, 1, og_shape[-1])
        quants = quants.reshape(*quants.shape[:-2], og_shape[-1], -1, block_size)
        quants = jnp.swapaxes(quants, -2, -3)
        quants = jnp.swapaxes(quants, -1, -2)
        scales = scales.reshape(*scales.shape[:-1], og_shape[-1], -1, 1)
        scales = jnp.swapaxes(scales, -2, -3)
        scales = jnp.swapaxes(scales, -1, -2)
        assert (quants.shape[-3] * quants.shape[-2]) == og_shape[
            -2
        ], f"{key}: {quants.shape} != {og_shape}"
        quant = QuantMatrix(
            shape=og_shape,
            dtype=og_dtype,
            quants=quants,
            scales=scales,
            use_approx=True,
            use_kernel=True,
            orig_dtype=og_tensor.dtype,
            mesh_and_axis=None,
        )
        array_flux[key] = quant
    flux = array_flux

    logger.info("Splitting linear1/linear2")
    flux = {k.replace("norm.scale", "norm.weight"): v for k, v in flux.items()}
    linear1 = flux.pop("single_blocks.linear1.weight")
    qkv, mlp = (
        weight_slice(linear1, axis=-1, start=0, size=model.config.hidden_size * 3),
        weight_slice(
            linear1,
            axis=-1,
            start=model.config.hidden_size * 3,
            size=model.config.mlp_size,
        ),
    )
    flux["single_blocks.attn.qkv_proj.weight"] = qkv
    flux["single_blocks.mlp.in_proj.weight"] = mlp
    linear1 = flux.pop("single_blocks.linear1.bias")
    qkv, mlp = (
        linear1[..., : model.config.hidden_size * 3],
        linear1[..., model.config.hidden_size * 3 :],
    )
    flux["single_blocks.attn.qkv_proj.bias"] = qkv
    flux["single_blocks.mlp.in_proj.bias"] = mlp
    linear2 = flux.pop("single_blocks.linear2.weight")
    qkv, mlp = (
        weight_slice(linear2, axis=1, start=0, size=model.config.hidden_size),
        weight_slice(
            linear2,
            axis=1,
            start=model.config.hidden_size,
            size=model.config.mlp_size,
        ),
    )
    flux["single_blocks.attn.o_proj.weight"] = qkv
    flux["single_blocks.mlp.out_proj.weight"] = mlp
    linear2 = flux.pop("single_blocks.linear2.bias")
    flux["single_blocks.attn.o_proj.bias"] = linear2
    flux["single_blocks.mlp.out_proj.bias"] = linear2 * 0
    norm_keys = [key for key in flux if key.startswith("single_blocks.norm")]
    for key in norm_keys:
        flux[key.replace("single_blocks.norm", "single_blocks.attn.qk_norm")] = (
            flux.pop(key)
        )
    return flux


def save_quantized(a):
    vals, treedef = jax.tree.flatten(
        a, is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue)
    )
    treedef = jax.tree.unflatten(treedef, [jnp.zeros((1,)) for _ in vals])
    new_vals = []
    for val in vals:
        if isinstance(val, QuantMatrix):
            new_vals.append(("qm", val.quants, val.scales, val.use_approx, str(val.orig_dtype), val.use_kernel, val.mesh_and_axis))
        else:
            new_vals.append(val)
    vals = new_vals
    return treedef, vals

def load_quantized(restored_treedef, restored_vals):
    _, restored_treedef = jax.tree.flatten(
        restored_treedef, is_leaf=lambda x: isinstance(x, qax.primitives.ArrayValue)
    )
    new_vals = []
    for val in restored_vals:
        if isinstance(val, list) and val[0] == "qm":
            quants, scales, use_approx, orig_dtype, use_kernel, mesh_and_axis = val[1:]
            new_vals.append(QuantMatrix(
                quants=quants, scales=scales,
                use_approx=use_approx, orig_dtype=orig_dtype,
                use_kernel=use_kernel, mesh_and_axis=mesh_and_axis))
        else:
            new_vals.append(val)
    restored_a = jax.tree.unflatten(restored_treedef, new_vals)
    return restored_a


def restore_array(path):
    checkpointer = ocp.PyTreeCheckpointer()
    vals_meta = checkpointer.metadata(path)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        vals_meta,
        sharding_tree=jax.tree.map(lambda _: sharding, vals_meta)
    )
    restored_vals = checkpointer.restore(path, restore_args=restore_args)
    return restored_vals


def load_flux(model, path="somewhere/flux.st", preprocess_into="somewhere/flux_prep"):
    logger.info(f"Loading flux from {path}")
    preprocess_into = Path(preprocess_into).resolve() if preprocess_into else None
    if preprocess_into is None or not preprocess_into.exists():
        flux = load_file(path)
        flux = preprocess_weights(flux, model)
        if preprocess_into is not None:
            logger.info("Saving preprocessed flux")
            flux_treedef, flux_vals = save_quantized(flux)
            path = ocp.test_utils.erase_and_create_empty(preprocess_into)
            checkpointer = ocp.PyTreeCheckpointer()
            try:
                checkpointer.save(path / "treedef", flux_treedef)
                checkpointer.save(path / "vals", flux_vals)
            except ValueError:
                shutil.rmtree(preprocess_into)
                raise
    else:
        logger.info("Loading preprocessed flux")
        flux_treedef = restore_array(preprocess_into / "treedef")
        flux_vals = restore_array(preprocess_into / "vals")
        flux = load_quantized(flux_treedef, flux_vals)

    # load weights
    logger.info("Loading weights")
    for key, value in flux.items():
        def replace_fn(old):
            assert old.shape == value.shape, f"{key}: {old.shape} != {value.shape}"
            v = value
            if not isinstance(v, QuantMatrix):
                v = v.astype(old.dtype)
            return v
        try:
            model = eqx.tree_at(selector_fn(key), model, replace_fn=replace_fn)
        except ValueError as e:
            raise ValueError(f"Error at {key}") from e
        except AttributeError as e:
            raise AttributeError(f"Error at {key}") from e

    return model


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
UINT_DTYPES = {
    8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64}

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

if __name__ == "__main__":
    logger.info("Creating inputs")
    
    batch_dims = (4, 4)

    key = jax.random.PRNGKey(0)
    dtype = jnp.bfloat16

    vae = FluxVAE(
        "somewhere/taef1/taef1_encoder.onnx", "somewhere/taef1/taef1_decoder.onnx"
    )

    import requests
    from PIL import Image
    dog_image_url = "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
    dog_image = Image.open(requests.get(dog_image_url, stream=True).raw)
    encoded = vae.encode(vae.preprocess(dog_image))
    if encoded.shape[-1] % 2:
        encoded = jnp.pad(encoded, ((0, 0), (0, 0), (0, 0), (0, 1)))
    if encoded.shape[-2] % 2:
        encoded = jnp.pad(encoded, ((0, 0), (0, 0), (0, 1), (0, 0)))

    import torch
    text_encodings = torch.load("somewhere/res.pt", map_location=torch.device("cpu"), weights_only=True)
    t5_emb, clip_emb = (x.detach().cpu().float().numpy().astype(ml_dtypes.bfloat16) for x in text_encodings[:2])
    n_seq_txt = t5_emb.shape[1]

    with jax.default_device(jax.devices("cpu")[0]):
        logger.info("Creating model")
        model = DiFormer.from_pretrained()

    def pad_to_batch(x):
        return jnp.repeat(x, np.prod(batch_dims), axis=0).reshape(
            *batch_dims, *x.shape[1:]
        )

    timesteps = jnp.full((1,), 0.1, dtype=jnp.float32)
    noise = jax.random.normal(key, encoded.shape, dtype=dtype)
    t = timesteps[..., None, None, None]
    noised = encoded * (1 - t) + noise * t
    timesteps = pad_to_batch(timesteps)
    txt = pad_to_batch(jnp.asarray(t5_emb, dtype))

    patched = rearrange(noised, "b c (h ph) (w pw) -> b h w (c ph pw)", ph=2, pw=2)
    h, w = patched.shape[-3:-1]
    patched = rearrange(patched, "b h w c -> b (h w) c")
    n_seq = patched.shape[-2]
    patched = pad_to_batch(patched)
    img = jnp.astype(patched, dtype)

    img_ids = jnp.zeros((*batch_dims, h, w, 3), dtype=jnp.uint32)
    img_ids = img_ids.at[..., 1].add(jnp.arange(h)[:, None])
    img_ids = img_ids.at[..., 2].add(jnp.arange(w)[None, :])
    img_ids = img_ids.reshape(*batch_dims, -1, 3)

    guidance_scale = jnp.full((*batch_dims,), 1, dtype=dtype)
    y = pad_to_batch(clip_emb)

    weights, logic = eqx.partition(model, eqx.is_array)

    @jax.jit
    @qax.use_implicit_args
    def f(weights, img, txt, timesteps, y, img_ids, guidance_scale):
        model = eqx.combine(weights, logic)
        return model(img=img, txt=txt, timesteps=timesteps, y=y, img_ids=img_ids, guidance=guidance_scale)

    logger.info("Loading model into mesh")
    shape_request = (-1, 2, 1)
    device_count = jax.device_count("tpu")
    mesh_shape = np.arange(device_count).reshape(*shape_request).shape
    physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(physical_mesh, ("dp", "fsdp", "tp"))
    mesh_and_axis = (mesh, None)  # FSDP
    def to_mesh(arr):
        if not is_arr(arr):
            return arr
        if isinstance(arr, QuantMatrix):
            return arr.with_mesh_and_axis(mesh_and_axis)
        return jax.device_put(arr,
                              jax.sharding.NamedSharding(
                                  mesh,
                                  jax.sharding.PartitionSpec(*((None,) * (arr.ndim)))))
    weights = jax.tree.map(to_mesh, weights, is_leaf=is_arr)

    logger.info("Loading inputs into mesh")
    args = img, txt, timesteps, y, img_ids, guidance_scale
    args = [
        jax.device_put(
            arg,
            jax.sharding.NamedSharding(mesh,
                                       jax.sharding.PartitionSpec(
                                           "dp", "fsdp", *((None,) * (arg.ndim - 2)))))
        for arg in args]

    logger.info("Running model")
    output = f(weights, *args)
    prediction = rearrange(output, "b d (h w) (c ph pw) -> (b d) c (h ph) (w pw)",
                          ph=2, pw=2, h=h, w=w)[0, :1]
    denoised = noised - t * prediction
    print(jnp.mean(jnp.abs(prediction)), jnp.mean(jnp.abs(encoded)), jnp.mean(jnp.abs(denoised)))
    generated = vae.deprocess(vae.decode(denoised))
    generated.save("somewhere/denoised.jpg")
