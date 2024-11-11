import sys
import numpy as np
import torch
sys.path.append("flux_orig/src")
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
torch.set_default_device("cpu")
cpu_model = load_flow_model("flux-dev", device="cpu")
image_inputs = torch.load("somewhere/image_inputs.pt", map_location=torch.device("cpu"))
text_inputs = torch.load("somewhere/text_inputs.pt", map_location=torch.device("cpu"))
reaped = torch.load("somewhere/reaped.pt", map_location=torch.device("cpu"))
print(image_inputs.keys())
encoded = image_inputs["encoded"]
h = encoded.shape[-2] // 2
w = encoded.shape[-1] // 2
t = image_inputs["timesteps"][..., None, None, None]
noise = image_inputs["noise"]
noised =  encoded * (1 - t) + noise * t
from einops import rearrange
patched = rearrange(noised, "... c (h ph) (w pw) -> ... (h w) (c ph pw)", ph=2, pw=2)
height, width = h, w


batch_dims = encoded.shape[:-3]
img_ids = torch.zeros((*batch_dims, h, w, 3), dtype=torch.int32)
img_ids[..., 1] += torch.arange(h)[:, None]
img_ids[..., 2] += torch.arange(w)[None, :]
img_ids = img_ids.reshape(*batch_dims, -1, 3)

img_torch = patched
guidance_vec = torch.full(
    (img_torch.shape[0],), 3.5, device=img_torch.device, dtype=img_torch.dtype
)
t_vec = torch.full(
    (img_torch.shape[0],), 0.5, dtype=img_torch.dtype, device=img_torch.device
)

print({k: v.shape for k, v in text_inputs.items()})
txt = text_inputs["txt"]
y = text_inputs["vec_in"]
n_seq_txt = txt.shape[-2]

txt_ids = torch.zeros((img_torch.shape[0], n_seq_txt, 3), device=img_torch.device, dtype=torch.int32)
v = lambda x: x.to(torch.bfloat16)
with torch.inference_mode():
    from collections import Counter
    n_occurred = Counter()
    for name, x in cpu_model(
        img=v(img_torch),
        img_ids=img_ids,
        txt=v(txt),
        txt_ids=txt_ids,
        y=v(y),
        timesteps=v(t_vec),
        guidance=v(guidance_vec),
    ):
        for key, value in x.items():
            key = f"{name}.{key}"
            try:
                old_value = reaped[key]
            except KeyError:
                print("warning: missing", key, "(shape:", value.shape, "norm:", str(value.abs().mean()) + ")")
                continue
            if old_value.shape[1:] == value.shape:
                layer_index = n_occurred.get(key, 0)
                print(key, "layer index", layer_index)
                old_value = old_value[layer_index]
            print(key, value.shape, old_value.shape)
            print(key, value.abs().mean(), old_value.abs().mean(), (old_value - value).abs().mean())
            n_occurred[key] += 1
x = x["img"]
x = unpack(x.float(), h * 16, w * 16)
denoised = noised - 0.5 * x
# generated = vae.deprocess(vae.decode(denoised))
# generated.save("somewhere/denoised_torch.jpg")