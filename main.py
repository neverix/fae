import os
os.environ["JAX_PLATFORMS"] = "cpu"
from fasthtml.common import FastHTML, serve
from fasthtml.common import FileResponse, JSONResponse
from src.vaov.vae import FluxVAE
from src.vaov.sae_common import SAEConfig, nf4
from src.vaov.scored_storage import ScoredStorage
import numpy as np
from pathlib import Path
import shutil

CACHE_DIRECTORY = "somewhere/maxacts"
vae = FluxVAE("somewhere/taef1/taef1_encoder.onnx", "somewhere/taef1/taef1_decoder.onnx")
cache_dir = Path(CACHE_DIRECTORY)
image_cache_dir = Path("somewhere/img_cache")
if image_cache_dir.exists():
    shutil.rmtree(image_cache_dir)
image_cache_dir.mkdir(parents=True, exist_ok=True)
scored_storage = ScoredStorage(
    cache_dir / "feature_acts.db",
    4, SAEConfig.top_k_activations,
    mode="r"
)
app = FastHTML()

@app.get("/cached_image/{step}/{image_id}")
def cached_image(step: int, image_id: int):
    img_path = image_cache_dir / f"{step}_{image_id}.jpg"
    if not img_path.exists():
        imgs_path = cache_dir / "images" / f"{step}.npz"
        if not imgs_path.exists():
            return {"error": "Image not found"}, 404
        imgs = np.load(imgs_path)["arr_0"]
        img = imgs[image_id:image_id+1]
        img = np.stack((img & 0x0F, (img & 0xF0) >> 4), -1).reshape(*img.shape[:-1], -1)
        img = nf4[img]
        img = img * SAEConfig.image_max
        img = vae.deprocess(vae.decode(img))
        img.save(img_path)
    return FileResponse(img_path)

@app.get("/feature_counts")
def feature_counts():
    counts = scored_storage.key_counts()
    counts = {key: int(val) for key, val in enumerate(counts)}
    return JSONResponse(counts)

@app.get("/maxacts/{feature_id}")
def maxacts(feature_id: int):
    rows = scored_storage.get_rows(feature_id)
    return JSONResponse(rows)

@app.get("/")
def home():
    return "<h1>Hello, World</h1>"

serve()
