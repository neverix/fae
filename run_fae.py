#!/usr/bin/env python
import os
import argparse
import shutil
from pathlib import Path
import numpy as np
import time
import traceback

# Import after setting up arguments to avoid immediate loading
def main():
    parser = argparse.ArgumentParser(description="Run the FAE visualization tool with configurable parameters")
    parser.add_argument("--cache-path", type=str, default="somewhere/maxacts_double_l18_img",
                        help="Path to the cache directory containing feature activations")
    parser.add_argument("--width", type=int, default=512,
                        help="Default width for generated images")
    parser.add_argument("--height", type=int, default=512,
                        help="Default height for generated images")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--clear-image-cache", action="store_true",
                        help="Clear the image cache before starting")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only compute and output metrics without starting the server")
    
    args = parser.parse_args()
    
    # Now import the rest after parsing arguments
    from fasthtml.common import fast_app, serve
    from fasthtml.common import FileResponse, JSONResponse
    from fasthtml.common import (
        Img, Div, Card, P, Table, Tbody, Tr, Td, A, H1, H2, Br,
        Form, Button, Input)
    from src.fae.vae import FluxVAE
    from src.fae.sae_common import SAEConfig, nf4
    from src.fae.scored_storage import ScoredStorage
    import requests
    from fh_plotly import plotly_headers, plotly2fasthtml
    import plotly.express as px
    
    # Set global variables based on arguments
    CACHE_DIRECTORY = args.cache_path
    DEFAULT_WIDTH = args.width
    DEFAULT_HEIGHT = args.height
    WIDTH = DEFAULT_WIDTH
    HEIGHT = DEFAULT_HEIGHT
    
    # Setup paths
    cache_dir = Path(CACHE_DIRECTORY)
    image_activations_dir = cache_dir / "image_activations"
    
    # Clear image cache if requested
    image_cache_dir = Path("somewhere/img_cache")
    if args.clear_image_cache and image_cache_dir.exists():
        shutil.rmtree(image_cache_dir)
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize storage
    if os.path.exists(cache_dir / "feature_acts.db") or True:
        while True:
            try:
                scored_storage = ScoredStorage(
                    cache_dir / "feature_acts.db",
                    3, SAEConfig.top_k_activations,
                    mode="r", use_backup=True
                )
                break
            except Exception as e:
                print(f"Error opening database: {e}")
                time.sleep(1)
    
    # Cache for spatial metrics
    spatial_metrics_cache = {}
    
    # Compute spatial metrics for a feature
    def compute_spatial_metrics(feature_id):
        rows = scored_storage.get_rows(feature_id)
        if not rows:
            return None
        
        # Extract positions and scores
        positions = []
        scores = []
        for (idx, h, w), score in rows:
            positions.append((h, w))
            scores.append(score)
        
        positions = np.array(positions)
        scores = np.array(scores)
        
        # Calculate center of mass
        total_score = np.sum(scores)
        if total_score == 0:
            return None
        
        center_h = np.sum(positions[:, 0] * scores) / total_score
        center_w = np.sum(positions[:, 1] * scores) / total_score
        
        # Calculate spatial spread (standard deviation from center)
        distances = np.sqrt((positions[:, 0] - center_h)**2 + (positions[:, 1] - center_w)**2)
        spatial_spread = np.sum(distances * scores) / total_score
        
        # Calculate concentration ratio (percentage of total activation in top 10% of positions)
        sorted_indices = np.argsort(scores)[::-1]
        top_10_percent = int(len(scores) * 0.1) or 1
        concentration_ratio = np.sum(scores[sorted_indices[:top_10_percent]]) / total_score
        
        # Calculate activation area (percentage of grid cells with significant activation)
        significant_threshold = 0.1 * np.max(scores)
        activation_area = np.sum(scores > significant_threshold) / len(scores)
        
        return {
            "center": (center_h, center_w),
            "spatial_spread": spatial_spread,
            "concentration_ratio": concentration_ratio,
            "activation_area": activation_area
        }
    
    # If metrics-only mode is enabled, compute and output metrics without starting the server
    if args.metrics_only:
        # Compute spatial sparsity
        def compute_spatial_sparsity():
            non_sparse_features = np.zeros(len(scored_storage), dtype=bool)
            img_list = list(image_activations_dir.glob("*.npz"))
            for img in img_list:
                saved = np.load(img)
                ind, wei = saved["arr_0"].ravel(), saved["arr_1"].ravel()
                non_sparse_features[ind] = True
            return 1 - non_sparse_features.mean()
        
        sparsity = compute_spatial_sparsity()
        print(f"Spatial Sparsity: {sparsity:.3f}")
        
        # Compute feature counts and maxima
        counts = scored_storage.key_counts()
        maxima = scored_storage.key_maxima()
        frequencies = counts.astype(np.float64) / counts.sum()
        
        # Compute metrics for all features
        all_metrics = {}
        for feature_id in range(len(scored_storage)):
            if counts[feature_id] > 0:
                metrics = compute_spatial_metrics(feature_id)
                if metrics:
                    all_metrics[feature_id] = metrics
        
        # Output summary statistics
        print(f"Total features: {len(scored_storage)}")
        print(f"Features with activations: {len(all_metrics)}")
        
        avg_spread = np.mean([m['spatial_spread'] for m in all_metrics.values()])
        avg_concentration = np.mean([m['concentration_ratio'] for m in all_metrics.values()])
        avg_area = np.mean([m['activation_area'] for m in all_metrics.values()])
        
        print(f"Average spatial spread: {avg_spread:.3f}")
        print(f"Average concentration ratio: {avg_concentration:.3f}")
        print(f"Average activation area: {avg_area:.3f}")
        
        return
    
    # Import the route decorator after setting up the app
    rt = fast_app.route
    
    @rt("/spatial_sparsity")
    def spatial_sparsity():
        non_sparse_features = np.zeros(len(scored_storage), dtype=bool)
        img_list = list(image_activations_dir.glob("*.npz"))
        for img in img_list:
            saved = np.load(img)
            ind, wei = saved["arr_0"].ravel(), saved["arr_1"].ravel()
            non_sparse_features[ind] = True
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
        batch_numbers = [int(img.stem) for img in img_list]
        frequencies = counts.astype(np.float64) / len(batch_numbers)
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
            try:
                full_activations = np.load(image_activations_dir / f"{idx}.npz")
                # Create a visualization of the activations
                # This is a simplified version - you might want to enhance this
                img_path = f"somewhere/img_cache/{feature_id}_{idx}.png"
                # Save the visualization as an image
                # For now, we'll just use a placeholder
                imgs.append(
                    Card(
                        P(f"Image {idx}, Max Activation: {grid.max():.3f}"),
                        style="margin: 10px; padding: 10px; border: 1px solid #ccc;"
                    )
                )
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
        
        return Div(
            H1(f"Max Activations for Feature {feature_id}"),
            P(metrics_display),
            P(A("Back to Top Features", href="/top_features")),
            Div(*imgs, style="display: flex; flex-wrap: wrap;"),
            style="padding: 20px;"
        )
    
    @rt("/top_features")
    def top_features():
        counts = scored_storage.key_counts()
        maxima = scored_storage.key_maxima()
        frequencies = counts.astype(np.float64) / counts.sum()
        
        cond = maxima > 4  # Threshold for selecting features
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
    
    @rt("/spatial_metrics")
    def spatial_metrics_view():
        # Get all feature IDs
        feature_ids = range(len(scored_storage))
        
        # Compute metrics for each feature
        all_metrics = {}
        for feature_id in feature_ids:
            if feature_id not in spatial_metrics_cache:
                spatial_metrics_cache[feature_id] = compute_spatial_metrics(feature_id)
            metrics = spatial_metrics_cache[feature_id]
            if metrics:
                all_metrics[feature_id] = metrics
        
        # Sort features by spatial spread
        sorted_features = sorted(all_metrics.items(), key=lambda x: x[1]['spatial_spread'])
        
        # Create table rows for display
        rows = []
        for feature_id, metrics in sorted_features:
            rows.append(
                Tr(
                    Td(str(feature_id)),
                    Td(f"{metrics['spatial_spread']:.3f}"),
                    Td(f"{metrics['concentration_ratio']:.3f}"),
                    Td(f"{metrics['activation_area']:.3f}"),
                    Td(A("View", href=f"/maxacts/{feature_id}"))
                )
            )
        
        return Div(
            H1("Spatial Metrics"),
            P("This table shows spatial metrics for all features with activations."),
            P(A("Back to Top Features", href="/top_features")),
            Table(
                Tbody(*rows),
                style="border-collapse: collapse; width: 100%;"
            ),
            style="padding: 20px;"
        )
    
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
    
    @rt("/gen_image", methods=["GET"])
    def gen_image():
        return Div(
            H1("Generate an image"),
            Form(
                P("Prompt:"),
                Input(name="prompt", type="text", value="a photo of a cat"),
                P("Width:"),
                Input(name="width", type="number", value=str(DEFAULT_WIDTH)),
                P("Height:"),
                Input(name="height", type="number", value=str(DEFAULT_HEIGHT)),
                Button("Generate", type="submit"),
                action="/generate",
                method="post"
            ),
            style="padding: 5em"
        )
    
    # Start the server
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Using cache directory: {CACHE_DIRECTORY}")
    serve(host=args.host, port=args.port)

if __name__ == "__main__":
    main() 
