import jax
import equinox as eqx
from dataclasses import replace
from typing import List
import jax.numpy as jnp
import numpy as np
from PIL import Image
from .clip import CLIPInterface
from .t5 import T5EncoderInferencer
from .flux_inferencer import FluxInferencer, random_or
from .diflayers import DiFormerConfig
from oryx.core.interpreters.harvest import sow, call_and_reap
from jax.experimental import mesh_utils
from loguru import logger

class FluxEnsemble:
    def __init__(
        self,
        use_schnell=True,
        use_fsdp=False,
        diformer_kwargs: dict = None,
        clip_name=None,
        t5_name=None,
    ):
        self.use_schnell = use_schnell
        curve_schedule = True
        if use_schnell:
            curve_schedule = False
            if diformer_kwargs is None:
                diformer_kwargs = {}
            diformer_kwargs["hf_path"] = (
                "black-forest-labs/FLUX.1-schnell",
                "flux1-schnell.safetensors",
            )
            diformer_kwargs["custom_config"] = replace(
                DiFormerConfig(),
                guidance_embed=False
            )
        logger.info("Creating mesh")
        self.curve_schedule = curve_schedule
        if use_fsdp:
            local_device_count = jax.local_device_count()
            shape_request = (-1, local_device_count, 1)
        else:
            shape_request = (-1, 1, 1)
        device_count = jax.device_count()
        mesh_shape = np.arange(device_count).reshape(*shape_request).shape
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
        self.mesh = jax.sharding.Mesh(physical_mesh, ("dp", "fsdp", "tp"))

        logger.info("Creating CLIP")
        self.clip = CLIPInterface(self.mesh, clip_name=clip_name)
        logger.info("Creating T5")
        self.t5 = T5EncoderInferencer(self.mesh, model_name=t5_name)
        logger.info("Creating Flux")
        if diformer_kwargs is None:
            diformer_kwargs = {}
        self.flux = FluxInferencer(self.mesh, diformer_kwargs=diformer_kwargs)

    def prepare_stuff(self, texts, key=None, width=None, height=None, images=None,
                      image_input_kwargs={}):
        batch_size = len(texts)
        clip_outputs = self.clip(texts)
        t5_outputs = self.t5(texts)
        text_input = self.flux.text_input(
            clip_emb=clip_outputs, t5_emb=t5_outputs, t5_already_sharded=True, clip_already_sharded=True
        )
        if images is None:
            prototype = Image.new("RGB", (width, height), (127, 127, 127))
            images = [prototype] * batch_size
        key = random_or(key)
        image_input_kwargs = dict(
            timesteps=1, guidance_scale=4.0, key=key
        ) | image_input_kwargs
        image_input = self.flux.image_input(images, **image_input_kwargs)
        return text_input, image_input

    def sample(self, texts: List[str], width: int = 512, height: int = 512,
               sample_steps: int = 1, debug_mode=False, decode_latents=True,
               key=None):
        n_tokens = width * height / (16 * 16)
        schedule = get_flux_schedule(n_tokens, sample_steps,
                                     shift_time=self.curve_schedule)

        logger.info("Sampling image for texts: {}", texts)
        batch_size = len(texts)
        logger.info("Preparing inputs")
        text_input, image_input = self.prepare_stuff(texts, width=width, height=height, key=key)
        logger.info("Sampling image")
        result = sample_jit(self.flux, text_input, image_input, schedule, debug_mode=debug_mode)
        # man, this would be cleaner as a monad
        if debug_mode:
            denoised, reaped = result
        else:
            denoised = result
        assert isinstance(denoised, jnp.ndarray)
        logger.info("Decoding")
        vae = self.flux.vae
        denoised = denoised.reshape(-1, *denoised.shape[2:])
        if decode_latents:
            decoded = np.asarray(vae.decode(denoised))
            images = [vae.deprocess(decoded[i:i+1]) for i in range(denoised.shape[0])]
        else:
            images = jax.device_put(denoised, jax.devices("cpu")[0])
        if debug_mode:
            return images, reaped
        else:
            return images


@eqx.filter_jit
def sample_jit(flux, text_input, image_input, schedule, debug_mode=False):
    schedule = schedule[:-1], schedule[1:]
    def sample_step(image_input, timesteps):
        more_noisy, less_noisy = timesteps
        flux_result = flux(text_input, image_input, debug_mode=debug_mode)
        if debug_mode:
            for k, v in flux_result.reaped.items():
                sow(v, name=k, tag="reaped", mode="clobber")
        return flux_result.next_input(more_noisy - less_noisy), None
    def generate(image_input):
        return jax.lax.scan(sample_step, image_input, schedule)[0].encoded
    result, reaped = call_and_reap(generate, tag="reaped")(image_input)
    if debug_mode:
        return result, reaped
    return result


def get_flux_schedule(n_seq: int, n_steps: int, shift_time=True):
    # https://github.com/black-forest-labs/flux/blob/478338d52759f92af9eeb92cc9eaa49582b20c78/src/flux/sampling.py#L78
    schedule = jnp.linspace(1, 0, n_steps + 1)
    if shift_time:
        mu = mu_estimator(n_seq)
        schedule = time_shift(mu, 1.0, schedule)
    return schedule


def time_shift(mu: float, sigma: float, t: jnp.ndarray):
    # that's (1 / t) - 1, not 1 / (t - 1)
    return jnp.exp(mu) / (jnp.exp(mu) + ((1 / t) - 1) ** sigma)


MAX_SEQ = 4096
MIN_SEQ = 256
MAX_SHIFT = 1.15
MIN_SHIFT = 0.5
def mu_estimator(n_seq: int):
    return n_seq * (MAX_SHIFT - MIN_SHIFT) / (MAX_SEQ - MIN_SEQ) + MIN_SHIFT


if __name__ == "__main__":
    ensemble = FluxEnsemble()
    for i, img in enumerate(ensemble.sample(["A mystic cat with a sign that says hello world!"] * 4)):
        img.save(f"somewhere/sample_{i}.png")
