from jaxonnxruntime import backend as jax_backend
from PIL import Image
import numpy as np
import requests
import onnx

# apt install git-lfs
# mkdir -p somewhere
# cd somewhere && git lfs clone https://huggingface.co/nev/taef1
dog_image_url = "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
dog_image = Image.open(requests.get(dog_image_url, stream=True).raw)
encoder_model = onnx.load("somewhere/taef1/taef1_encoder.onnx")
encoder_rep = jax_backend.BackendRep(encoder_model)
decoder_model = onnx.load("somewhere/taef1/taef1_decoder.onnx")
decoder_rep = jax_backend.BackendRep(decoder_model)
inputs = (
    np.array(dog_image).astype(np.float32)[None, :, :, :].transpose(0, 3, 1, 2) / 255.0
)
print("running encoder")
encoded = encoder_rep.run({"input": inputs})
print("running decoder")
decoded = decoder_rep.run({"input": encoded[0]})
output = Image.fromarray(
    ((decoded[0].transpose(0, 2, 3, 1)) * 255.0).clip(0, 255).astype(np.uint8)[0]
).convert("RGB")
output.save("somewhere/dog.jpg")
