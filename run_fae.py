#!/usr/bin/env python
import os
import argparse
import shutil
from pathlib import Path
import numpy as np
import time
import traceback
import json

# Import after setting up arguments to avoid immediate loading
def main():
    parser = argparse.ArgumentParser(description="Run the FAE visualization tool with configurable parameters")
    parser.add_argument("--cache-path", type=str, default="somewhere/maxacts_itda_50k_256/itda_new_data",
                        help="Path to the cache directory containing feature activations")
    parser.add_argument("--width", type=int, default=16,
                        help="Default width for generated images")
    parser.add_argument("--height", type=int, default=16,
                        help="Default height for generated images")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--clear-image-cache", action="store_true",
                        help="Clear the image cache before starting")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only compute and output metrics without starting the server")
    parser.add_argument("--json-output", type=str,
                        help="Path to save metrics as JSON (only used with --metrics-only)")
    parser.add_argument("--json-pretty", action="store_true",
                        help="Pretty-print the JSON output")
    parser.add_argument("--sample-size", type=int, default=-1,
                        help="Only process a sample of features (-1 means process all features)")
    
    args = parser.parse_args()
    
    # Now import the rest after parsing arguments
    from src.fae.vae import FluxVAE
    from src.fae.sae_common import SAEConfig, nf4
    from src.fae.scored_storage import ScoredStorage
    import requests
    import plotly.express as px
    
    # Set global variables based on arguments
    CACHE_DIRECTORY = args.cache_path
    DEFAULT_WIDTH = args.width
    DEFAULT_HEIGHT = args.height
    WIDTH = DEFAULT_WIDTH
    HEIGHT = DEFAULT_HEIGHT
    
    # Setup paths
    cache_dir = Path(CACHE_DIRECTORY)
    image_activations_dir = Path("somewhere/maxacts_itda_50k_256/image_activations_itda_new")
    
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
        
        # Calculate concentration ratio (percentage of total activation in top 25% of positions)
        sorted_indices = np.argsort(scores)[::-1]
        top_25_percent = int(len(scores) * 0.25) or 1
        concentration_ratio = np.sum(scores[sorted_indices[:top_25_percent]]) / total_score
        
        # Calculate activation area (percentage of grid cells with activation above a threshold)
        threshold = 0.3
        activation_area = np.sum(scores > threshold) / len(scores)

        # Print the feature id and the metrics
        print(f"Feature {feature_id}:")
        print(f"  Spatial Spread: {spatial_spread:.3f}")
        print(f"  Concentration Ratio: {concentration_ratio:.3f}")
        print(f"  Activation Area: {activation_area:.3f}")

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
                feature_counts = np.bincount(ind[wei > 0.0], minlength=len(scored_storage))
                non_sparse_features |= feature_counts > 6
            return 1 - non_sparse_features.mean()
        
        sparsity = compute_spatial_sparsity()
        print(f"Spatial Sparsity: {sparsity:.3f}")
        
        # Compute feature counts and maxima
        counts = scored_storage.key_counts()
        maxima = scored_storage.key_maxima()
        frequencies = counts.astype(np.float64) / counts.sum()
        
        # Compute metrics for features
        all_metrics = {}
        # Process all features or just a sample based on command-line argument
        feature_count = len(scored_storage)
        if args.sample_size > 0:
            feature_count = min(args.sample_size, feature_count)
            print(f"Processing sample of {feature_count} features")
        
        for feature_id in range(feature_count):
            if counts[feature_id] >= 6:
                metrics = compute_spatial_metrics(feature_id)
                if metrics:
                    # Convert numpy values to Python native types for JSON serialization
                    metrics_dict = {
                        "center": (float(metrics["center"][0]), float(metrics["center"][1])),
                        "spatial_spread": float(metrics["spatial_spread"]),
                        "concentration_ratio": float(metrics["concentration_ratio"]),
                        "activation_area": float(metrics["activation_area"]),
                        "frequency": float(frequencies[feature_id]),
                        "count": int(counts[feature_id]),
                        "max_activation": float(maxima[feature_id])
                    }
                    all_metrics[feature_id] = metrics_dict
        
        # Calculate summary statistics
        spatial_spreads = np.array([m['spatial_spread'] for m in all_metrics.values()])
        avg_spread = np.mean(spatial_spreads)
        var_spread = np.var(spatial_spreads)
        index_of_dispersion = var_spread / avg_spread if avg_spread > 0 else 0
        
        avg_concentration = np.mean([m['concentration_ratio'] for m in all_metrics.values()])
        avg_area = np.mean([m['activation_area'] for m in all_metrics.values()])
        
        # Output summary statistics
        print(f"Total features: {len(scored_storage)}")
        print(f"Features with activations: {len(all_metrics)}")
        print(f"Average spatial spread: {avg_spread:.3f}")
        print(f"Index of dispersion: {index_of_dispersion:.3f}")
        print(f"Average concentration ratio: {avg_concentration:.3f}")
        print(f"Average activation area: {avg_area:.3f}")
        
        # Prepare JSON output
        metrics_json = {
            "summary": {
                "total_features": len(scored_storage),
                "features_with_activations": len(all_metrics),
                "spatial_sparsity": float(sparsity),
                "avg_spatial_spread": float(avg_spread),
                "index_of_dispersion": float(index_of_dispersion),
                "avg_concentration_ratio": float(avg_concentration),
                "avg_activation_area": float(avg_area)
            },
            "features": all_metrics
        }
        
        # Save JSON output if requested
        if args.json_output:
            json_path = Path(args.json_output)
            # Create parent directories if they don't exist
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON to file
            with open(json_path, 'w') as f:
                if args.json_pretty:
                    json.dump(metrics_json, f, indent=2)
                else:
                    json.dump(metrics_json, f)
            print(f"Metrics saved to {json_path}")
        
        return
    
if __name__ == "__main__":
    main()
