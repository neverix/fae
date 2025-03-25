import os
os.environ["JAX_PLATFORMS"] = "cpu"
from fasthtml.common import fast_app, serve
from fasthtml.common import FileResponse, JSONResponse
from fasthtml.common import (
    Img, Div, Card, P, Table, Tbody, Tr, Td, A, H1, H2, Br,
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

CACHE_DIRECTORY = "somewhere/maxacts_double_l18_img"
HEIGHT, WIDTH = 16, 16
vae = FluxVAE("somewhere/taef1/taef1_encoder.onnx", "somewhere/taef1/taef1_decoder.onnx")
cache_dir = Path(CACHE_DIRECTORY)
image_activations_dir = cache_dir / "image_activations" 
image_cache_dir = Path("somewhere/img_cache")
if image_cache_dir.exists():
    shutil.rmtree(image_cache_dir)
image_cache_dir.mkdir(parents=True, exist_ok=True)
if os.path.exists(cache_dir / "feature_acts.db") or True:
    while True:
        try:
            scored_storage = ScoredStorage(
                cache_dir / "feature_acts.db",
                3, SAEConfig.top_k_activations,
                mode="r", use_backup=True
            )
        except (ValueError, EOFError):
            traceback.print_exc()
            time.sleep(0.01)
            continue
        break
app, rt = fast_app(hdrs=plotly_headers)


# Add a function to compute spatial metrics for a feature
def compute_spatial_metrics(feature_id):
    """Compute spatial metrics for a specific feature."""
    rows = scored_storage.get_rows(feature_id)
    
    # Group rows by idx
    metrics_by_image = {}
    for (idx, h, w), score in rows:
        key = idx
        if key not in metrics_by_image:
            # Create activation grid for this image
            grid = np.zeros((HEIGHT, WIDTH), dtype=float)
            metrics_by_image[key] = {"grid": grid, "activations": []}
        
        # Add score to the grid
        metrics_by_image[key]["grid"][h, w] = score
        metrics_by_image[key]["activations"].append((h, w, score))
    
    # Compute metrics for each image
    results = {}
    for idx, data in metrics_by_image.items():
        grid = data["grid"]
        
        # Skip if no activations
        if grid.sum() == 0:
            continue
            
        # Get positions where activation occurs
        active_positions = np.where(grid > 0)
        if len(active_positions[0]) == 0:
            continue
            
        # Compute center of mass
        h_indices, w_indices = np.indices((HEIGHT, WIDTH))
        total_activation = grid.sum()
        center_h = np.sum(h_indices * grid) / total_activation if total_activation > 0 else 0
        center_w = np.sum(w_indices * grid) / total_activation if total_activation > 0 else 0
        
        # Compute average distance from center of mass (spatial spread)
        distances = np.sqrt((h_indices - center_h)**2 + (w_indices - center_w)**2)
        avg_distance = np.sum(distances * grid) / total_activation if total_activation > 0 else 0
        
        # Compute concentration ratio: what percentage of total activation is in the top 25% of active pixels
        active_values = grid[active_positions]
        sorted_values = np.sort(active_values)[::-1]  # Sort in descending order
        quarter_point = max(1, len(sorted_values) // 4)
        concentration_ratio = np.sum(sorted_values[:quarter_point]) / total_activation if total_activation > 0 else 0
        
        # Compute activation area: percentage of image area that has activations
        activation_area = len(active_positions[0]) / (HEIGHT * WIDTH)
        
        # Store metrics
        results[idx] = {
            "spatial_spread": float(avg_distance),
            "concentration_ratio": float(concentration_ratio),
            "activation_area": float(activation_area),
            "max_activation": float(grid.max()),
            "center": (float(center_h), float(center_w))
        }
    
    # Aggregate metrics across images
    if results:
        avg_metrics = {
            "spatial_spread": float(np.mean([m["spatial_spread"] for m in results.values()])),
            "concentration_ratio": float(np.mean([m["concentration_ratio"] for m in results.values()])),
            "activation_area": float(np.mean([m["activation_area"] for m in results.values()])),
            "num_images": len(results)
        }
        return avg_metrics
    return None

# Cache for spatial metrics to avoid recomputation
spatial_metrics_cache = {}

@rt("/cached_image/{image_id}")
def cached_image(image_id: int):
    img_path = image_cache_dir / f"{image_id}.jpg"
    if not img_path.exists():
        imgs_path = cache_dir / "images" / f"{image_id}.npz"
        if not imgs_path.exists():
            return {"error": "Image not found"}, 404
        img = np.load(imgs_path)["arr_0"][None]
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
        Br(),
        H1(f"Spatial sparsity: {spatial_sparsity():.3f}"),
        Br(),
        P(A("View Spatial Metrics", href="/spatial_metrics")),
        Br(),
        *[Card(
            P(f"Feature {i}, Frequency: {frequencies[i]:.5f}, Max: {maxima[i]}"),
            A("View Max Acts", href=f"/maxacts/{i}")
        ) for i in top_few],
        style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; padding: 20px;"
    )

@rt("/spatial_sparsity")
def spatial_sparsity():
    non_sparse_features = np.zeros(len(scored_storage), dtype=bool)
    img_list = list(image_activations_dir.glob("*.npz"))
    for img in img_list:
        saved = np.load(img)
        ind, wei = saved["arr_0"].ravel(), saved["arr_1"].ravel()
        feature_counts = np.bincount(ind[wei > 0.0], minlength=len(scored_storage))
        non_sparse_features |= feature_counts > 6
    return 1 - non_sparse_features.mean()


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

    # Group rows by idx
    grouped_rows = {}
    for (idx, h, w), score in rows:
        key = idx
        if key not in grouped_rows:
            grouped_rows[key] = np.zeros((HEIGHT, WIDTH), dtype=float)

        # Add score to the corresponding location in the grid
        grouped_rows[key][h, w] = score

    # Compute spatial metrics for this feature if not already cached
    if feature_id not in spatial_metrics_cache:
        spatial_metrics_cache[feature_id] = compute_spatial_metrics(feature_id)
    
    metrics = spatial_metrics_cache[feature_id]
    metrics_display = ""
    if metrics:
        metrics_display = f"Spatial Spread: {metrics['spatial_spread']:.3f}, Concentration: {metrics['concentration_ratio']:.3f}, Active Area: {metrics['activation_area']:.3f}"
    
    # Prepare images and cards
    imgs = []
    for idx, grid in sorted(grouped_rows.items(), key=lambda x: x[1].max(), reverse=True)[:20]:
        full_activations = np.load(image_activations_dir / f"{idx}.npz")
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
                cell_content = ""
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
            Img(src=f"/cached_image/{idx}", style="width: 100%; height: auto; position: relative;"),
            heatmap_table,
            style="position: relative; width: 300px; height: 300px; overflow: hidden;"
        )

        # Add to images
        imgs.append(Card(
            Div(
                P(f"Index: {idx}, Score: {grid.max()}"),
                overlaid_image
            )
        ))

    return Div(
        P(A("<- Go back", href="/top_features")),
        H2(f"Feature {feature_id} Spatial Metrics: {metrics_display}"),
        Div(*imgs, style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center"),
        style="padding: 20px"
    )

# Add a new endpoint to view spatial metrics for all features
@rt("/spatial_metrics")
def spatial_metrics_view():
    # Get all feature IDs
    counts = scored_storage.key_counts()
    maxima = scored_storage.key_maxima()
    
    # Filter features with significant activations
    cond = maxima > 4
    features = np.arange(len(scored_storage))[cond]
    
    # Compute metrics for all features (with caching)
    all_metrics = []
    for feature_id in features:
        if feature_id not in spatial_metrics_cache:
            spatial_metrics_cache[feature_id] = compute_spatial_metrics(feature_id)
        
        metrics = spatial_metrics_cache[feature_id]
        if metrics:
            all_metrics.append({
                "feature_id": int(feature_id),
                "spatial_spread": metrics["spatial_spread"],
                "concentration_ratio": metrics["concentration_ratio"],
                "activation_area": metrics["activation_area"],
                "num_images": metrics["num_images"]
            })
    
    # Sort by activation area (from most concentrated to most dispersed)
    all_metrics.sort(key=lambda x: x["activation_area"])
    
    # Create scatter plot of concentration vs spatial spread
    scatter_plot = plotly2fasthtml(px.scatter(
        x=[m["activation_area"] for m in all_metrics],
        y=[m["concentration_ratio"] for m in all_metrics],
        hover_name=[f"Feature {m['feature_id']}" for m in all_metrics],
        labels={"x": "Activation Area (% of image)", "y": "Concentration Ratio"},
        title="Spatial Concentration Analysis"
    ))
    
    # Create cards for features
    feature_cards = [
        Card(
            P(f"Feature {m['feature_id']}"),
            P(f"Concentration: {m['concentration_ratio']:.3f}"),
            P(f"Active Area: {m['activation_area']:.3f}%"),
            P(f"Spatial Spread: {m['spatial_spread']:.3f}"),
            A("View Max Acts", href=f"/maxacts/{m['feature_id']}"),
            style="width: 200px; margin: 10px;"
        ) for m in all_metrics[:50]  # Show top 50 most concentrated features
    ]
    
    return Div(
        H1("Spatial Metrics Analysis"),
        P(A("<- Go back", href="/top_features")),
        Br(),
        scatter_plot,
        Br(),
        H2("Most Concentrated Features (Lowest Activation Area)"),
        Div(*feature_cards, style="display: flex; flex-wrap: wrap; justify-content: center;"),
        style="padding: 20px;"
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
        response = requests.post("http://localhost:8000/sample", json={"prompts": prompts, "sample_steps": 20})
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
        P(A("Spatial Metrics", href="/spatial_metrics")),
        P(A("Generator", href="/gen_image")),
        style="padding: 5em"
    )

serve()