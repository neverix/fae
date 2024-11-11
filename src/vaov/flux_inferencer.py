import random
from dataclasses import asdict
from functools import partial
import torch
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from .quant import QuantMatrix
import fire
from einops import rearrange
from loguru import logger
from .diformer import DiFormer, is_arr
import equinox as eqx
from .diflayers import global_mesh
import qax
from oryx.core.interpreters.harvest import call_and_reap
from typing import Optional
from jax.experimental import mesh_utils
import ml_dtypes
from .vae import FluxVAE


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
    reaped: Optional[dict]

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


@partial(jax.jit, static_argnums=(1,), static_argnames=("debug_mode",))
@qax.use_implicit_args
def run_model_(weights, logic, kwargs, debug_mode=False):
    model = eqx.combine(weights, logic)
    results = call_and_reap(model, tag="debug")(**kwargs)
    if debug_mode:
        return results
    return results[0]


class DiFormerInferencer:
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
        self.vae = FluxVAE(*vae_args)

        with jax.default_device(jax.devices("cpu")[0]):
            logger.info("Creating model")
            model = DiFormer.from_pretrained(**diformer_kwargs)

            logger.info("Loading model into mesh")

        global_mesh.mesh = mesh
        mesh_and_axis = (mesh, None)  # FSDP

        logger.info("Moving model to device")
        weights, logic = eqx.partition(model, eqx.is_array)

        def to_mesh(arr):
            if not is_arr(arr):
                return arr
            if isinstance(arr, QuantMatrix):
                return arr.with_mesh_and_axis(mesh_and_axis)
            return jax.device_put(
                arr,
                jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec(*((None,) * (arr.ndim)))
                ),
            )

        weights = jax.tree.map(to_mesh, weights, is_leaf=is_arr)
        self.weights, self.logic = weights, logic

    def __call__(self, text_inputs, image_inputs, debug_mode=False):
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
        results = run_model_(self.weights, self.logic, kwargs, debug_mode=debug_mode)
        reaped = None
        if debug_mode:
            patched, reaped = results
        else:
            patched = results
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

    def image_input(self, images, timesteps=0, guidance_scale=3.5, key=None):
        # fixme: OOM otherwise
        with jax.default_device(jax.devices("cpu")[0]):
            encoded = self.vae.encode(
                jnp.concatenate([self.vae.preprocess(image) for image in images], 0)
            )
        encoded = encoded.astype(jnp.float32)
        if encoded.shape[-1] % 2:
            encoded = jnp.pad(encoded, ((0, 0), (0, 0), (0, 0), (0, 1)))
        if encoded.shape[-2] % 2:
            encoded = jnp.pad(encoded, ((0, 0), (0, 0), (0, 1), (0, 0)))

        batch = encoded.shape[0]
        if isinstance(timesteps, (int, float)):
            timesteps = jnp.full((batch,), timesteps, dtype=jnp.float32)
        key = random_or(key)
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
    shape_request = (-1, 1, 1)
    device_count = jax.device_count()
    mesh_shape = np.arange(device_count).reshape(*shape_request).shape
    physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(physical_mesh, ("dp", "fsdp", "tp"))

    logger.info("Creating inferencer")
    inferencer = DiFormerInferencer(mesh)
    logger.info("Creating inputs")
    # batch_size = device_count
    batch_size = 32
    image_inputs = inferencer.image_input(
        [dog_image] * batch_size, timesteps=0.5, key=jax.random.key(1)
    )
    text_inputs = inferencer.text_input(
        t5_emb.repeat(batch_size, 1, 1), clip_emb.repeat(batch_size, 1)
    )
    logger.info("Warming up model")
    result = jax.block_until_ready(inferencer(text_inputs, image_inputs))
    logger.info("Running model")
    result = jax.block_until_ready(inferencer(text_inputs, image_inputs))
    logger.info("Running model for debug")
    result = jax.block_until_ready(inferencer(text_inputs, image_inputs, debug_mode=True))

    logger.info("Comparing results")
    pred_noise = result.noise[0, :1]
    noise = image_inputs.noise[0, :1]
    print(
        jnp.mean(jnp.square(noise)),
        jnp.mean(jnp.square(pred_noise)),
        jnp.mean(jnp.square(noise - pred_noise)),
    )
    print(jnp.mean(jnp.square(result.prediction - result.ground_truth)))
    print(
        jnp.mean(
            jnp.square(
                # jax.random.normal(jax.random.key(0), result.prediction.shape)
                (-image_inputs.encoded)
                - result.ground_truth
            )
        )
    )
    print({k: v.shape for k, v in result.reaped.items()})

    def tt(v):
        if isinstance(v, jnp.ndarray):
            return torch.from_numpy(np.asarray(v[0, :1].astype(jnp.float32)))
        return v

    torch.save({k: tt(v) for k, v in result.reaped.items()}, "somewhere/reaped.pt")
    torch.save(
        {k: tt(v) for k, v in image_inputs.to_dict().items()},
        "somewhere/image_inputs.pt",
    )
    torch.save({k: tt(v) for k, v in text_inputs.items()}, "somewhere/text_inputs.pt")

    logger.info("Saving results")
    vae = inferencer.vae
    denoised = result.denoised[0, :1]
    generated = vae.deprocess(vae.decode(denoised))
    generated.save("somewhere/denoised.jpg")


if __name__ == "__main__":
    fire.Fire(main)
