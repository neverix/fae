from jaxonnxruntime import backend as jax_backend
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from PIL import Image
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import onnx


class FluxVAE(eqx.Module):
    encoder_rep: jax_backend.BackendRep
    decoder_rep: jax_backend.BackendRep

    def __init__(self, encoder_model_path, decoder_model_path):
        self.encoder_rep = jax_backend.BackendRep(onnx.load(encoder_model_path))
        self.decoder_rep = jax_backend.BackendRep(onnx.load(decoder_model_path))

    def preprocess(self, image):
        inputs = (
            np.array(image).astype(np.float32)[None, :, :, :].transpose(0, 3, 1, 2)
            / 255.0
        )
        return inputs

    @eqx.filter_jit
    def encode(self, inputs):
        return self.encoder_rep.run({"input": inputs})[0]

    @eqx.filter_jit
    def decode(self, encoded):
        return self.decoder_rep.run({"input": encoded})[0]

    def deprocess(self, decoded):
        return Image.fromarray(
            ((np.asarray(decoded).transpose(0, 2, 3, 1)) * 255.0)
            .clip(0, 255)
            .astype(np.uint8)[0]
        ).convert("RGB")


class FluxVAEHQ(eqx.Module):
    vae: AutoencoderKL = eqx.field(static=True)
    processor: VaeImageProcessor = eqx.field(static=True)


    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
        self.processor = VaeImageProcessor(
            vae_scale_factor=(2 ** (len(self.vae.config.block_out_channels) - 1)) * 2,
        )

    def preprocess(self, image):
        return jnp.asarray(self.processor.preprocess(image).numpy())

    @torch.no_grad
    def encode(self, inputs):
        inputs = torch.from_numpy(np.asarray(inputs))
        encoded = self.vae.encode(inputs, return_dict=False)[0].mean
        encoded = (encoded - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return jnp.asarray(encoded.numpy())

    @torch.no_grad
    def decode(self, encoded):
        encoded = torch.from_numpy(np.asarray(encoded))
        encoded = (encoded / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        decoded = self.vae.decode(encoded, return_dict=False)[0]
        return jnp.asarray(decoded.numpy())

    def deprocess(self, decoded):
        decoded = torch.from_numpy(np.asarray(decoded))
        return self.processor.postprocess(decoded)[0]


# apt install git-lfs
# mkdir -p somewhere
# cd somewhere && git lfs clone https://huggingface.co/nev/taef1
# uv pip install -U ./jort && rm -rf .venv && uv sync && JAX_PLATFORMS=cpu uv run python -m src.fae.vae
if __name__ == "__main__":
    from PIL import Image
    import requests
    dog_image_url = "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
    dog_image = Image.open(requests.get(dog_image_url, stream=True).raw)
    # vae = FluxVAE("somewhere/taef1/taef1_encoder.onnx", "somewhere/taef1/taef1_decoder.onnx")
    vae = FluxVAEHQ()
    output = vae.deprocess(vae.decode(vae.encode(vae.preprocess(dog_image))))
    output.save("somewhere/dog.jpg")
