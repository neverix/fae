import os
os.environ["JAX_PLATFORMS"] = "cpu"
from fasthtml.common import fast_app, serve
from fasthtml.common import FileResponse, JSONResponse
from fasthtml.common import (
    Img, Div, Card, P, Table, Tbody, Tr, Td, A, H1, H2,
    Form, Button, Input)
from src.fae.vae import FluxVAE
from src.fae.sae_common import SAEConfig, nf4
from src.fae.scored_storage import ScoredStorage
import numpy as np
from pathlib import Path
import shutil
import requests
from fh_plotly import plotly_headers, plotly2fasthtml
import plotly.express as px
import traceback
import time

CACHE_DIRECTORY = "somewhere/maxacts"
HEIGHT, WIDTH = 16, 16
vae = FluxVAE("somewhere/taef1/taef1_encoder.onnx", "somewhere/taef1/taef1_decoder.onnx")
cache_dir = Path(CACHE_DIRECTORY)
image_activations_dir = cache_dir / "image_activations" 
image_cache_dir = Path("somewhere/img_cache")
if image_cache_dir.exists():
    shutil.rmtree(image_cache_dir)
image_cache_dir.mkdir(parents=True, exist_ok=True)
while True:
    try:
        scored_storage = ScoredStorage(
            cache_dir / "feature_acts.db",
            4, SAEConfig.top_k_activations,
            mode="r", use_backup=True
        )
    except (ValueError, EOFError) as e:
        traceback.print_exc()
        time.sleep(0.01)
        continue
    break
app, rt = fast_app(hdrs=plotly_headers)

@rt("/cached_image/{step}/{image_id}")
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


@rt("/top_features")
def top_features():
    counts = scored_storage.key_counts()
    maxima = scored_storage.key_maxima()
    frequencies = counts.astype(np.float64) / counts.sum()
    # expected_frequency = 4 / counts.size
    # metric = np.abs(frequencies - expected_frequency)
    # metric[maxima < 5] = np.inf
    # correct_order = np.argsort(metric)
    # matches = np.arange(len(scored_storage))[maxima > 3.5]
    cond = maxima > 4  # 4 for single/18, 3 for double/18
    # cond = maxima > 2
    # cond = frequencies > 5e-5
    # cond &= frequencies < 0.0031
    matches = np.arange(len(scored_storage))[cond]
    correct_order = np.random.permutation(matches)
    top_few = correct_order[:256].tolist()
    return Div(
        H1(f"Top features ({len(matches)}/{len(matches) / len(scored_storage) * 100:.2f}% match criteria)"),
        *[Card(
            P(f"Feature {i}, Frequency: {frequencies[i]:.5f}, Max: {maxima[i]}"),
            A("View Max Acts", href=f"/maxacts/{i}")
        ) for i in top_few],
        style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; padding: 20px;"
    )

@rt("/feature_counts")
def feature_counts():
    counts = scored_storage.key_counts()
    counts = {key: int(val) for key, val in enumerate(counts)}
    return JSONResponse(counts)

@rt("/fry_plot")
def fry_plot():
    counts = scored_storage.key_counts()
    maxima = scored_storage.key_maxima()
    img_list = list(image_activations_dir.glob("*.npz"))
    batch_numbers = [int(img.stem.partition("_")[0]) for img in img_list]
    seq_numbers = [int(img.stem.split("_")[1]) for img in img_list]
    frequencies = counts.astype(np.float64) / (max(batch_numbers) * (max(seq_numbers) + 1))
    return plotly2fasthtml(px.scatter(
        x=frequencies,
        y=maxima,
        labels={"x": "Frequency", "y": "Max Activation"},
        title="Fry Plot"
    ))

@rt("/maxacts/{feature_id}")
def maxacts(feature_id: int):
    rows = scored_storage.get_rows(feature_id)

    # Group rows by (step, idx)
    grouped_rows = {}
    for (step, idx, h, w), score in rows:
        key = (step, idx)
        if key not in grouped_rows:
            grouped_rows[key] = np.zeros((HEIGHT, WIDTH), dtype=float)

        # Add score to the corresponding location in the grid
        grouped_rows[key][h, w] = score

    # Prepare images and cards
    imgs = []
    for (step, idx), grid in sorted(grouped_rows.items(), key=lambda x: x[1].max(), reverse=True)[:20]:
        full_activations = np.load(image_activations_dir / f"{step}_{idx}.npz")
        gravel = grid.ravel()
        k = full_activations["arr_0"].shape[1]
        for i, (f, w) in enumerate(zip(full_activations["arr_0"].ravel(), full_activations["arr_1"].ravel())):
            if f == feature_id:
                gravel[i // k] = w

        # Normalize the grid for color intensity
        normalized_grid = (grid - grid.min()) / (grid.max() - grid.min()) if grid.max() > grid.min() else grid

        # Create a heatmap table
        heatmap_rows = []
        for row in range(grid.shape[0]):
            td_cells = []
            for col in range(grid.shape[1]):
                score = grid[row, col]
                norm_value = normalized_grid[row, col]

                # Calculate color intensity (semi-transparent blue)
                blue_intensity = int(255 * norm_value)
                color = f"rgba(0, 0, 255, {0.5 * norm_value})"

                # Create cell with background color and score
                # cell_content = f"{score:.2f}"
                cell_content = f""
                td_cell = Td(cell_content,
                             style=f"background-color: {color}; text-align: center; padding: 1px; color: white; font-size: 1px;")
                td_cells.append(td_cell)

            heatmap_row = Tr(*td_cells)
            heatmap_rows.append(heatmap_row)

        # Compile the heatmap
        heatmap_table = Table(
            Tbody(*heatmap_rows),
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-collapse: collapse; pointer-events: none;"
        )

        # Create a container for overlaying the heatmap on the image
        overlaid_image = Div(
            Img(src=f"/cached_image/{step}/{idx}", style="width: 100%; height: auto; position: relative;"),
            heatmap_table,
            style="position: relative; width: 300px; height: 300px; overflow: hidden;"
        )

        # Add to images
        imgs.append(Card(
            Div(
                P(f"Step: {step}, Index: {idx}, Score: {grid.max()}"),
                overlaid_image
            )
        ))

    return Div(
        P(A("<- Go back", href="/top_features")),
        Div(*imgs, style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center"),
        style="padding: 20px"
    )

NUM_PROMPTS = 4

@rt("/gen_image", methods=["GET"])
def gen_image():
    prompt_inputs = [
        Input(type="text", name=f"prompt-{i}", placeholder=f"Enter prompt {i+1}", style="width: 100%; margin-bottom: 10px;", value="cat")
        for i in range(NUM_PROMPTS)
    ]

    return Div(
        H1("Image Generation"),
        H2("Enter Prompts:"),
        Form(
            *prompt_inputs,
            Button("Generate Images", type="button", hx_post="/generate", hx_target="#image-results", hx_indicator="#loading"),
            method="POST" # still needed to pass the data
        ),
        Div(id="loading", style="display:none;", children=[P("Generating...")]),
        Div(id="image-results"),
        style="padding: 20px;"
    )

@rt("/generate", methods=["POST"])
def generate(form: dict):
    prompts = [form.get(f"prompt-{i}", "") for i in range(NUM_PROMPTS)]
    prompts = [p for p in prompts if p]
    images = []
    error_message = None

    if not prompts:
        return P("At least one prompt is required.", style="color: red;")

    try:
        response = requests.post("http://localhost:8000/sample", json={"prompts": prompts})
        response.raise_for_status()
        data = response.json()
        images = data["images"]
    except requests.exceptions.RequestException as e:
        return P(f"Error generating images: {e}", style="color: red;")

    image_elements = [
        Img(src=f"data:image/png;base64,{img}", style="max-width: 300px; max-height: 300px; margin: 10px;")
        for img in images
    ]
    return Div(*image_elements)


@rt("/")
def home():
    return Div(
        H1("fae"),
        H2("SAE"),
        P(A("Top features", href="/top_features")),
        P(A("Generator", href="/gen_image")),
        style="padding: 5em"
    )

serve()
