# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
import equinox as eqx
from equinox import nn
import jax.experimental
import jax.experimental.shard_map
from safetensors.numpy import load_file
from functools import partial
from collections import defaultdict
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float, UInt
from typing import Optional
from .quant import QuantMatrix
from .quant_loading import load_thing, save_thing
import qax.primitives
import qax
import jax.numpy as jnp
import jax
from pathlib import Path
from loguru import logger
from typing import TypeVar, Generic
from .diflayers import (
    SingleStreamBlock, DoubleStreamBlock, LastLayer, timestep_embedding, EmbedND, MLPEmbedder,
    VLinear, DiFormerConfig, sow_debug, timestep_embedding)
from .dumb_rng import dumb_prng_impl



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

    def call_first(self, x, *args, **kwargs):
        layer = eqx.combine(jax.tree.map(lambda x: x[0], self.weights), self.logic)
        return layer(x, *args, **kwargs)

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
        sow_debug(dict(img=img, txt=txt, vec=vec, pe=pe, mask=mask), "pre_double")
        
        # is not actually called if we don't reap
        first_double = self.double_blocks.call_first(
            data, vec=vec, pe=pe, mask=mask, debug_first=True
        )
        sow_debug(first_double, "first_double")
        
        data = self.double_blocks(data, vec=vec, pe=pe, mask=mask)

        txt, img = data["txt"], data["img"]
        data = jnp.concatenate((txt, img), -2)
        sow_debug(dict(data=data), "pre_single")
        data = self.single_blocks(data, vec=vec, pe=pe, mask=mask)
        img = data[..., txt.shape[-2]:, :]

        sow_debug(dict(data=img), "pre_final")
        img = self.final_layer(img, vec)[..., :orig_img_len, :]
        sow_debug(dict(img=img), "final_img")
        
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


def preprocess_official(flux, model):
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
        x = values[""]
        x = x.transpose(*range(0, x.ndim - 2), x.ndim - 1, x.ndim - 2)
        
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
        
        array_flux[key] = x
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


def load_flux(
    model,
    path=None,
    # path="somewhere/flux.st",
    hf_path=("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
    # preprocess_into="somewhere/flux_prep",
    preprocess_into=None,
):
    logger.info(f"Loading flux from {path}")
    preprocess_into = Path(preprocess_into).resolve() if preprocess_into else None
    if preprocess_into is None or not preprocess_into.exists():
        if path is not None:
            flux = load_file(path)
            flux = preprocess_weights(flux, model)
        else:
            flux = load_file(hf_hub_download(
                repo_id=hf_path[0], filename=hf_path[1],
            ))
            flux = preprocess_official(flux, model)
        if preprocess_into is not None:
            logger.info("Saving preprocessed flux")
            save_thing(flux, preprocess_into)
    else:
        logger.info("Loading preprocessed flux")
        flux = load_thing(preprocess_into)

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
