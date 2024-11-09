import jax
from typing import List
import jax.numpy as jnp
import numpy as np
from PIL import Image
from .clip import CLIPInterface
from .t5 import T5EncoderInferencer
from .diformer import DiFormerInferencer
from loguru import logger

class FluxEnsemble:
    def __init__(self):
        self.mesh = jax.sharding.Mesh(
            np.array(jax.devices("tpu")).reshape(-1, jax.local_device_count(), 1), ("dp", "fsdp", "tp")
        )
        self.clip = CLIPInterface(self.mesh)
        self.t5 = T5EncoderInferencer(self.mesh)
        self.flux = DiFormerInferencer(self.mesh)

    def sample(self, text: str, width: int = 512, height: int = 512, batch_size: int = 4,
               sample_steps: int = 200):
        logger.info("Sampling image for text: {}", text)
        texts = [text] * batch_size
        logger.info("Encoding text using CLIP")
        clip_outputs = self.clip(texts)
        logger.info("Encoding text using T5")
        t5_outputs = self.t5(texts)
        text_input = jax.block_until_ready(self.flux.text_input(
            clip_emb=clip_outputs, t5_emb=t5_outputs, t5_already_sharded=True, clip_already_sharded=True
        ))
        logger.warning("using something else anyway for debugging")
        import torch
        torch.set_grad_enabled(False)
        text_encodings = torch.load("somewhere/res.pt", map_location=torch.device("cpu"), weights_only=True)
        t5_emb, clip_emb = text_encodings[:2]
        text_input = self.flux.text_input(t5_emb.repeat(batch_size, 1, 1), clip_emb.repeat(batch_size, 1))
        logger.info("Preparing image input")
        prototype = Image.new("RGB", (width, height), (127, 127, 127))
        image_input = self.flux.image_input([prototype] * batch_size, timesteps=1)
        logger.info("Sampling image")
        def sample_step(_, image_input):
            return self.flux(text_input, image_input).next_input(1 / sample_steps)
        denoised = jax.block_until_ready(jax.lax.fori_loop(0, sample_steps, sample_step, image_input).noised)
        # ... step here to fully denoise the image
        logger.info("Decoding")
        vae = self.flux.vae
        denoised = denoised.reshape(-1, *denoised.shape[2:])
        decoded = np.asarray(vae.decode(denoised))
        for i in range(denoised.shape[0]):
            yield vae.deprocess(decoded[i:i+1])


if __name__ == "__main__":
    ensemble = FluxEnsemble()
    for i, img in enumerate(ensemble.sample("A mystic cat with a sign that says hello world!")):
        img.save(f"somewhere/sample_{i}.png")
