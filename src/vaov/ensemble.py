import jax
from typing import List
import jax.numpy as jnp
import numpy as np
from PIL import Image
from .clip import CLIPInterface
from .t5 import T5EncoderInferencer
from .flux_inferencer import DiFormerInferencer
from jax.experimental import mesh_utils
from loguru import logger

class FluxEnsemble:
    def __init__(self):
        logger.info("Creating mesh")
        local_device_count = jax.local_device_count()
        shape_request = (-1, local_device_count, 1)
        device_count = jax.device_count()
        mesh_shape = np.arange(device_count).reshape(*shape_request).shape
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
        self.mesh = jax.sharding.Mesh(physical_mesh, ("dp", "fsdp", "tp"))
        
        self.clip = CLIPInterface(self.mesh)
        self.t5 = T5EncoderInferencer(self.mesh)
        self.flux = DiFormerInferencer(self.mesh)

    def sample(self, texts: list[str], width: int = 512, height: int = 512,
               sample_steps: int = 20):
        n_tokens = width * height / (16 * 16)
        schedule = get_flux_schedule(n_tokens, sample_steps)
        
        logger.info("Sampling image for texts: {}", texts)
        batch_size = len(texts)
        logger.info("Encoding text using CLIP")
        clip_outputs = self.clip(texts)
        logger.info("Encoding text using T5")
        t5_outputs = self.t5(texts)
        text_input = jax.block_until_ready(self.flux.text_input(
            clip_emb=clip_outputs, t5_emb=t5_outputs, t5_already_sharded=True, clip_already_sharded=True
        ))
        logger.info("Preparing image input")
        prototype = Image.new("RGB", (width, height), (127, 127, 127))
        image_input = self.flux.image_input([prototype] * batch_size,
                                            timesteps=1, guidance_scale=4.0,
                                            key=jax.random.key(5))
        logger.info("Sampling image")
        def sample_step(image_input, timesteps):
            more_noisy, less_noisy = timesteps
            return self.flux(text_input, image_input).next_input(more_noisy - less_noisy), None
        schedule = (schedule[:-1], schedule[1:])
        denoised = jax.block_until_ready(jax.lax.scan(sample_step, image_input, schedule)[0].encoded)
        # ... step here to fully denoise the image
        logger.info("Decoding")
        vae = self.flux.vae
        denoised = denoised.reshape(-1, *denoised.shape[2:])
        decoded = np.asarray(vae.decode(denoised))
        for i in range(denoised.shape[0]):
            yield vae.deprocess(decoded[i:i+1])


def get_flux_schedule(n_seq: int, n_steps: int):
    # https://github.com/black-forest-labs/flux/blob/478338d52759f92af9eeb92cc9eaa49582b20c78/src/flux/sampling.py#L78
    schedule = jnp.linspace(1, 0, n_steps + 1)
    mu = mu_estimator(n_seq)
    timesteps = time_shift(mu, 1.0, schedule)
    return timesteps


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
    for i, img in enumerate(ensemble.sample("A mystic cat with a sign that says hello world!")):
        img.save(f"somewhere/sample_{i}.png")
