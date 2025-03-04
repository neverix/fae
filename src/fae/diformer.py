# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
import equinox as eqx
from equinox.internal._loop import scan as eqx_scan
from equinox import nn
import jax.experimental
import jax.experimental.shard_map
from safetensors.numpy import load_file
from functools import partial
from collections import defaultdict
from huggingface_hub import hf_hub_download
from jaxtyping import Array, Float, UInt
from typing import Optional
from .quant import MockQuantMatrix, is_arr
from .quant_loading import load_thing, save_thing
import jax.numpy as jnp
import jax
from pathlib import Path
from loguru import logger
from typing import TypeVar, Generic
from .diflayers import (
    SingleStreamBlock, DoubleStreamBlock, LastLayer, EmbedND, MLPEmbedder,
    VLinear, DiFormerConfig, timestep_embedding)
from .dumb_rng import dumb_prng_impl


def unify(arg, *args, repeat=None):
    if not is_arr(arg):
        return arg
    if isinstance(arg, MockQuantMatrix):
        if repeat is not None:
            args = (arg,) * (repeat - 1)
        return arg.stack(*args)
    if repeat is not None:
        return jnp.repeat(arg[None], repeat, axis=0)
    return jnp.stack((arg, *args), axis=0)


T = TypeVar('T', bound=eqx.Module)

class SequentialScan(eqx.Module, Generic[T]):
    layer: T
    n_layers: int = eqx.field(static=True)

    def __init__(self, layers: tuple[T, ...], repeat: int = None):
        self.layer = jax.tree.map(partial(unify, repeat=repeat), *layers, is_leaf=is_arr)
        self.n_layers = repeat

    # @eqx.filter_checkpoint
    def __call__(self, x, *args, **kwargs):
        pred = lambda x: is_arr(x) and x.ndim and x.shape[0] == self.n_layers
        weights, logic = eqx.partition(self.layer, pred)
        # weights_mock = MockQuantMatrix.mockify(weights)

        # def scan_fn(carry):
        #     carry, i = carry
        #     layer = jax.tree.map(lambda x: x[i] if is_arr(x) else x, layer_mock, is_leaf=is_arr)
        #     layer = MockQuantMatrix.unmockify(layer)
        #     return (layer(carry, *args, **kwargs, layer_idx=i), i + 1)

        # return while_loop(
        #     lambda *args, **kwargs: jnp.array(True),
        #     scan_fn,
        #     (x, jnp.array(0, dtype=jnp.uint32)),
        #     max_steps=self.n_layers,
        #     kind="checkpointed"
        # )[0]

        # @jax.remat
        def scan_fn(carry, weight: T):
            carry, i = carry
            # print("---", weight)
            # weight = MockQuantMatrix.unmockify(weight)
            # print("===", weight)
            layer = eqx.combine(weight, logic)
            return (layer(carry, *args, **kwargs, layer_idx=i), i + 1), None

        # TODO remove this
        if "MockQuantMatrix" in str(weights):
            return eqx_scan(scan_fn, (x, jnp.array([0], dtype=jnp.uint32)), weights, kind="checkpointed")[0][0]
        else:
            return jax.lax.scan(scan_fn, (x, jnp.array(0, dtype=jnp.uint32)), weights)[0][0]

    def __getattr__(self, name):
        return getattr(self.layer, name)


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
    config_cls: type = eqx.field(default=DiFormerConfig, static=True)

    config: DiFormerConfig = eqx.field(static=True)
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
        time_emb = timestep_embedding(timesteps, self.config.time_embed_dim)
        vec = self.time_in(time_emb)
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            g_emb = timestep_embedding(guidance.astype(jnp.bfloat16), self.config.guidance_embed_dim)
            vec = vec + self.guidance_in(g_emb)
        vec = vec + self.vector_in(y)
        vec = vec.astype(jnp.bfloat16)

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
        img = data[..., txt.shape[-2]:, :]

        img = self.final_layer(img, vec)[..., :orig_img_len, :]

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


@partial(jax.jit, static_argnames=("axis", "start", "size"))
def weight_slice(arr, *, axis: int, start: int, size: int):
    if isinstance(arr, MockQuantMatrix):
        return arr.slice(axis=axis, start=start, size=size)
    return jax.lax.dynamic_slice_in_dim(arr, start, size, axis=axis)


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
    flux = {k.replace("final_layer.adaLN_modulation.1.", "final_layer.adaLN_modulation."): v for k, v in flux.items()}

    # fold in nf4 arrays
    logger.info("Quantizing arrays")
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
        x = x.astype(jnp.bfloat16)
        if any(s in key for s in ("vector_in", "guidance_in")):
            array_flux[key] = x
            continue
        x = MockQuantMatrix.quantize(x, mode="nf4", group_size=64)
        array_flux[key] = x
    flux = array_flux

    flux = {k.replace("norm.scale", "norm.weight"): v for k, v in flux.items()}
    logger.info("Splitting linear1")
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
    logger.info("Splitting linear2")
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
    logger.info("Done preprocessing")
    return flux


def load_flux(
    model,
    path=None,
    # path="somewhere/flux.st",
    hf_path=("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
    preprocess_into="somewhere/flux_prep",
    # preprocess_into=None,
):
    logger.info(f"Loading flux from {path or hf_path}")
    preprocess_into = Path(preprocess_into).resolve() if preprocess_into else None
    preprocess_into = preprocess_into / "--".join(hf_path[0].split("/")) if preprocess_into else None
    if preprocess_into is None or not preprocess_into.exists():
        if path is not None:
            assert False, "nf4 file loading no longer supported"
            flux = load_file(path)
            flux = preprocess_weights(flux, model)
        else:
            flux = load_file(hf_hub_download(
                repo_id=hf_path[0], filename=hf_path[1],
            ))
            flux = preprocess_official(flux, model)
        if preprocess_into is not None:
            logger.info("Saving preprocessed flux")
            preprocess_into.mkdir(parents=True, exist_ok=True)
            save_thing(flux, preprocess_into)
        else:
            logger.info("Skipping save")
    else:
        logger.info("Loading preprocessed flux")
        flux = load_thing(preprocess_into)

    # load weights
    logger.info("Loading weights")
    for key, value in flux.items():
        def replace_fn(old):
            v = value
            if isinstance(v, dict):
                v = MockQuantMatrix(
                    **v,
                    orig_dtype=jnp.bfloat16,
                    use_approx=True,
                    use_kernel=True,
                    mesh_and_axis=None,
                )
            assert old.shape == v.shape, f"{key}: {old.shape} != {v.shape}"
            if not isinstance(v, MockQuantMatrix):
                v = v.astype(old.dtype)
            return v
        try:
            model = eqx.tree_at(selector_fn(key), model, replace_fn=replace_fn)
        except ValueError as e:
            raise ValueError(f"Error at {key}") from e
        except AttributeError as e:
            raise AttributeError(f"Error at {key}") from e

    return model
