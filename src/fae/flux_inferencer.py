import random
from dataclasses import asdict
from typing import Optional, Dict
import torch
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from .quant import MockQuantMatrix
import fire
from einops import rearrange
from loguru import logger
from .diformer import DiFormer, is_arr
import equinox as eqx
from .diflayers import global_mesh
from typing import Union
from jax.experimental import mesh_utils
import ml_dtypes
from .vae import FluxVAE, FluxVAEHQ
from .interp_globals import post_double_reaper, post_single_reaper


def random_or(key):
    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))
    return key


def from_torch(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        y = x.detach().cpu().numpy()
        if x.dtype == torch.bfloat16:
            y = y.astype(ml_dtypes.bfloat16)
        x = y
    if isinstance(x, np.ndarray):
        x = jnp.array(x)
    return x


class DotDict(eqx.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        return asdict(self)


class ImageInput(DotDict):
    encoded: Float[Array, "*batch c h w"]
    noise: Float[Array, "*batch c h w"]
    timesteps: Float[Array, "*batch"]
    guidance_scale: Float[Array, "*batch"]

    @property
    def noised(self):
        t = self.timesteps[..., None, None, None]
        return self.encoded * (1 - t) + self.noise * t

    @property
    def n_seq(self):
        return self.h * self.w

    @property
    def h(self):
        return self.encoded.shape[-2] // 2

    @property
    def w(self):
        return self.encoded.shape[-1] // 2

    @property
    def batch_dims(self):
        return self.encoded.shape[:-3]

    @property
    def patched(self):
        return rearrange(
            self.noised, "... c (h ph) (w pw) -> ... (h w) (c ph pw)", ph=2, pw=2
        )

    @property
    def img_ids(self):
        batch_dims, h, w = self.batch_dims, self.h, self.w
        img_ids = jnp.zeros((*batch_dims, h, w, 3), dtype=jnp.uint32)
        img_ids = img_ids.at[..., 1].add(jnp.arange(h)[:, None])
        img_ids = img_ids.at[..., 2].add(jnp.arange(w)[None, :])
        img_ids = img_ids.reshape(*batch_dims, -1, 3)
        return img_ids


class ImageOutput(DotDict):
    previous_input: ImageInput
    patched: Float[Array, "*batch n_seq (patch_size patch_size in_channels)"]
    reaped: Dict[str, Optional[jax.typing.ArrayLike]]

    @property
    def prediction(self):
        h, w = self.previous_input.h, self.previous_input.w
        prediction = rearrange(
            self.patched,
            "... (h w) (c ph pw) -> ... c (h ph) (w pw)",
            ph=2,
            pw=2,
            h=h,
            w=w,
        )
        return prediction

    @property
    def ground_truth(self):
        return self.previous_input.noise - self.previous_input.encoded

    @property
    def denoised(self):
        noised, timesteps = self.previous_input.noised, self.previous_input.timesteps
        prediction = self.prediction
        return noised - timesteps[..., None, None, None] * prediction

    @property
    def noise(self):
        noised, timesteps = self.previous_input.noised, self.previous_input.timesteps
        prediction = self.prediction
        return noised + (1 - timesteps[..., None, None, None]) * prediction

    def next_input(self, time: float):
        return ImageInput(
            encoded=self.denoised,
            noise=self.noise,
            timesteps=self.previous_input.timesteps - time,
            guidance_scale=self.previous_input.guidance_scale,
        )

    @property
    def loss(self):
        return jnp.mean(jnp.square(self.ground_truth - self.prediction))  # simple as that


call_simple = lambda arg, **kwargs: arg(**kwargs)
call_plain = eqx.filter_jit(call_simple)


class FluxInferencer(eqx.Module):
    mesh: jax.sharding.Mesh = eqx.field(static=True)
    vae: Union[FluxVAE, FluxVAEHQ]
    model: DiFormer

    def __init__(
        self,
        mesh,
        vae_args=(
            "somewhere/taef1/taef1_encoder.onnx",
            "somewhere/taef1/taef1_decoder.onnx",
        ),
        diformer_kwargs=None,
    ):
        if diformer_kwargs is None:
            diformer_kwargs = {}
        self.mesh = mesh

        logger.info("Creating VAE")
        if "hq" in vae_args:
            self.vae = FluxVAEHQ()
        else:
            self.vae = FluxVAE(*vae_args)

        with jax.default_device(jax.devices("cpu")[0]):
            logger.info("Creating model")
            model = DiFormer.from_pretrained(**diformer_kwargs)

            logger.info("Loading model into mesh")

        global_mesh.mesh = mesh
        mesh_and_axis = (mesh, None)  # FSDP

        logger.info("Moving model to device")

        def to_mesh(arr):
            if not is_arr(arr):
                return arr
            if isinstance(arr, MockQuantMatrix):
                return arr.with_mesh_and_axis(mesh_and_axis)
            return jax.device_put(
                arr,
                jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec(*((None,) * (arr.ndim)))
                ),
            )

        self.model = jax.tree.map(to_mesh, model, is_leaf=is_arr)

    def __call__(
        self,
        text_inputs,
        image_inputs,
        reap_double=[],
        reap_single=[],
    ):
        img, timesteps, img_ids, guidance_scale, h, w = [
            image_inputs[k]
            for k in ("patched", "timesteps", "img_ids", "guidance_scale", "h", "w")
        ]
        txt, vec_in = [text_inputs[k] for k in ("txt", "vec_in")]
        kwargs = dict(
            img=img,
            txt=txt,
            timesteps=timesteps,
            y=vec_in,
            img_ids=img_ids,
            guidance=guidance_scale,
        )
        patched, reaped = eqx.filter_jit(post_single_reaper.reap(
            post_double_reaper.reap(
                call_simple,
                restrict_to_layers=reap_double,
                no_reaped=True,),
            restrict_to_layers=reap_single))(self.model, **kwargs)
        return ImageOutput(previous_input=image_inputs, patched=patched, reaped=reaped)

    def to_mesh(self, x, already_sharded=False):
        def to_mesh_fn(x):
            if not isinstance(x, jnp.ndarray):
                return x
            if already_sharded:
                return x
            x = x.reshape(self.mesh.shape["dp"], -1, *x.shape[1:])
            return jax.device_put(
                x,
                jax.sharding.NamedSharding(
                    self.mesh,
                    jax.sharding.PartitionSpec("dp", "fsdp", *((None,) * (x.ndim - 2))),
                ),
            )

        return jax.tree.map(to_mesh_fn, x)

    def text_input(
        self, t5_emb, clip_emb, clip_already_sharded=False, t5_already_sharded=False
    ):
        txt = self.to_mesh(from_torch(t5_emb), already_sharded=t5_already_sharded)
        vec_in = self.to_mesh(
            from_torch(clip_emb), already_sharded=clip_already_sharded
        )
        return dict(
            txt=txt,
            vec_in=vec_in,
        )

    def image_input(self, images, timesteps=0, guidance_scale=3.5, key=None,
                    already_preprocessed=False, already_encoded=False):
        # fixme: OOM otherwise
        with jax.default_device(jax.devices("cpu")[0]):
            if not already_preprocessed:
                images = jnp.concatenate([self.vae.preprocess(image) for image in images], 0)
            if not already_encoded:
                encoded = self.vae.encode(images)
            else:
                encoded = images
        encoded = encoded.astype(jnp.float32)
        if encoded.shape[-1] % 2:
            encoded = jnp.pad(encoded, ((0, 0), (0, 0), (0, 0), (0, 1)))
        if encoded.shape[-2] % 2:
            encoded = jnp.pad(encoded, ((0, 0), (0, 0), (0, 1), (0, 0)))

        batch = encoded.shape[0]
        key = random_or(key)
        if isinstance(timesteps, (int, float)):
            timesteps = jnp.full((batch,), timesteps, dtype=jnp.float32)
        elif timesteps is None:
            timesteps = jax.random.uniform(key, (batch,), dtype=jnp.float32)
        noise = jax.random.normal(key, encoded.shape, dtype=encoded.dtype)

        if isinstance(guidance_scale, (int, float)):
            guidance_scale = jnp.full((batch,), guidance_scale, dtype=jnp.float32)

        return self.to_mesh(
            ImageInput(
                encoded=encoded,
                noise=noise,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
            )
        )


def main():
    import jax_smi
    jax_smi.initialise_tracking()

    logger.info("Creating inputs")

    import requests
    from PIL import Image

    dog_image_url = "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
    dog_image = Image.open(requests.get(dog_image_url, stream=True).raw)
    dog_image = dog_image.resize((1024, 720))

    torch.set_grad_enabled(False)
    text_encodings = torch.load(
        "somewhere/res.pt", map_location=torch.device("cpu"), weights_only=True
    )
    t5_emb, clip_emb = text_encodings[:2]

    logger.info("Creating mesh")
    # shape_request = (1, -1, 1)
    shape_request = (-1, jax.local_device_count(), 1)
    # shape_request = (-1, 1, 1)
    # shape_request = (2, 2, 1)
    device_count = jax.device_count()
    mesh_shape = np.arange(device_count).reshape(*shape_request).shape
    physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(physical_mesh, ("dp", "fsdp", "tp"))

    logger.info("Creating inferencer")
    inferencer = FluxInferencer(mesh)
    logger.info("Creating inputs")
    # batch_size = device_count
    batch_size = 16
    image_inputs = inferencer.image_input(
        [dog_image] * batch_size, timesteps=0.5, key=jax.random.key(1)
    )
    text_inputs = inferencer.text_input(
        t5_emb.repeat(batch_size, 1, 1), clip_emb.repeat(batch_size, 1)
    )
    logger.info("Warming up model")
    result = jax.block_until_ready(inferencer(text_inputs, image_inputs))
    logger.info("Running model")
    result = jax.block_until_ready(inferencer(text_inputs, image_inputs, reap_double=[10], reap_single=[20]))

    print({k: v.shape for k, v in result.reaped.items()})

    logger.info("Saving results")
    vae = inferencer.vae
    denoised = result.denoised[0, :1]
    generated = vae.deprocess(vae.decode(denoised))
    generated.save("somewhere/denoised.jpg")


if __name__ == "__main__":
    fire.Fire(main)
