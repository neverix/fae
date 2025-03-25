#!/usr/bin/env python
import os
import argparse
import shutil
from pathlib import Path
import numpy as np
import time
import traceback
import einops
import json
from scipy.spatial.distance import pdist  # Add to imports at top

# Import after setting up arguments to avoid immediate loading
def main():
    parser = argparse.ArgumentParser(description="Run the FAE visualization tool with configurable parameters")
    parser.add_argument("--cache-path", type=str, default="somewhere/maxacts_double_l12_img-k64",
                        help="Path to the cache directory containing feature activations")
    parser.add_argument("--width", type=int, default=16,
                        help="Default width for generated images")
    parser.add_argument("--height", type=int, default=16,
                        help="Default height for generated images")
    parser.add_argument("--clear-image-cache", action="store_true",
                        help="Clear the image cache before starting")
    parser.add_argument("--json-output", type=str,
                        help="Path to save metrics as JSON (only used with --metrics-only)")
    parser.add_argument("--json-pretty", action="store_true",
                        help="Pretty-print the JSON output")
    parser.add_argument("--sample-size", type=int, default=-1,
                        help="Only process a sample of features (-1 means process all features)")
    parser.add_argument("--activation-threshold", type=float, default=0.05,
                        help="Threshold for considering a feature as active on a position in an image")
    parser.add_argument("--activation-image-threshold", type=float, default=4.0,
                        help="Threshold for considering a feature as active (sum of all activations on an image)")
    parser.add_argument("--activated-positions-threshold", type=int, default=3,
                        help="Threshold for considering a feature as active (number of positions on an image with activations)")
    parser.add_argument("--feature-activated-images-threshold", type=int, default=6,
                        help="Threshold for considering a feature as active (number of images with activations)")
    args = parser.parse_args()
    
    # Now import the rest after parsing arguments
    from src.fae.vae import FluxVAE
    from src.fae.sae_common import SAEConfig, nf4
    from src.fae.scored_storage import ScoredStorage
    
    # Set global variables based on arguments
    CACHE_DIRECTORY = args.cache_path
    DEFAULT_WIDTH = args.width
    DEFAULT_HEIGHT = args.height
    WIDTH = DEFAULT_WIDTH
    HEIGHT = DEFAULT_HEIGHT
    ACTIVATION_THRESHOLD = args.activation_threshold
    ACTIVATED_POSITIONS_THRESHOLD = args.activated_positions_threshold
    FEATURE_ACTIVATED_IMAGES_THRESHOLD = args.feature_activated_images_threshold
    ACTIVATION_IMAGE_THRESHOLD = args.activation_image_threshold
    # Setup paths
    cache_dir = Path(CACHE_DIRECTORY)
    # Use the activations directory directly from the cache path
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

    # Compute spatial metrics for a feature
    def compute_spatial_metrics(feature_id):
        rows = scored_storage.get_rows(feature_id)
        if not rows:
            print(f"No rows found for feature {feature_id}")
            return None
        
        # Add accumulator for average activation pattern
        avg_activation_pattern = np.zeros((HEIGHT, WIDTH), dtype=float)
        image_count = 0
        
        # Group rows by idx
        metrics_by_image = {}
        for (idx, h, w), score in rows:
            key = idx
            if key not in metrics_by_image:
                grid = np.zeros((HEIGHT, WIDTH), dtype=float)
                metrics_by_image[key] = {"grid": grid, "activations": []}
            
            full_activations = np.load(image_activations_dir / f"{idx}.npz")
            gravel = grid.ravel()
            k = full_activations["arr_0"].shape[1]
            for i, (f, w) in enumerate(zip(full_activations["arr_0"].ravel(), full_activations["arr_1"].ravel())):
                if f == feature_id:
                    gravel[i // k] = w
            
            grid = gravel.reshape((HEIGHT, WIDTH))
            metrics_by_image[key]["grid"] = grid
            
            # Only include grids that meet activation thresholds
            if grid.sum() > ACTIVATION_IMAGE_THRESHOLD:
                active_positions = np.where(grid > ACTIVATION_THRESHOLD)
                if len(active_positions[0]) > ACTIVATED_POSITIONS_THRESHOLD:
                    avg_activation_pattern += (grid > ACTIVATION_THRESHOLD).astype(float)
                    image_count += 1

        # Calculate binary activation pattern and entropy
        if image_count > 0:
            avg_activation_pattern /= image_count
            binary_pattern = (avg_activation_pattern > 0.5).astype(float)
            
            # Get coordinates of activated points
            activated_coords = np.array(np.where(binary_pattern > 0)).T
            
            if len(activated_coords) > 1:
                # Calculate pairwise distances between all activated points
                distances = []
                for i in range(len(activated_coords)):
                    for j in range(i + 1, len(activated_coords)):
                        dist = np.sqrt(np.sum((activated_coords[i] - activated_coords[j]) ** 2))
                        distances.append(dist)
                avg_distance = np.mean(distances) if distances else 0.0
            else:
                avg_distance = 0.0

            # Rest of entropy calculation
            flat_pattern = binary_pattern.ravel()
            counts = np.bincount(flat_pattern.astype(int), minlength=2)
            probabilities = counts / len(flat_pattern)
            probabilities = probabilities[probabilities > 0]
            entropy = -np.sum(probabilities * np.log2(probabilities))
        else:
            entropy = 0.0
            binary_pattern = np.zeros((HEIGHT, WIDTH))
            avg_distance = 0.0

        # Compute metrics for each image
        results = {}
        for idx, data in metrics_by_image.items():
            grid = data["grid"]
            #print(f"Checking activations for image {idx} in feature {feature_id}")
            # Skip if activations are below threshold
            #print(f"Sum of activations: {grid.sum()}")
            if grid.sum() <= ACTIVATION_IMAGE_THRESHOLD:
               #print(f"Skipping image {idx} in feature {feature_id} because the sum of activations is below threshold")
                continue
            # Get positions where activation occurs
            active_positions = np.where(grid > ACTIVATION_THRESHOLD)
            #print(f"Active positions: {active_positions}")
            #print(f"Number of activated positions: {len(active_positions[0])}")
            if len(active_positions[0]) <= ACTIVATED_POSITIONS_THRESHOLD:
                #print(f"Skipping image {idx} in feature {feature_id} because it doesn't have enough activated positions")
                #print(f"Number of activated positions: {len(active_positions[0])}")
                #print(f"Threshold: {ACTIVATED_POSITIONS_THRESHOLD}")
                continue
            
            # print(f"Image {idx} in feature {feature_id} has enough activated positions")
            # Compute center of mass
            # get mean activation position
            h_indices, w_indices = np.indices((HEIGHT, WIDTH))
            total_activation = grid.sum()
            center_h = np.sum(h_indices * grid) / total_activation if total_activation > 0 else 0
            center_w = np.sum(w_indices * grid) / total_activation if total_activation > 0 else 0
            
            # Compute average distance from center of mass (spatial spread)
            #distances = np.sqrt((h_indices - center_h)**2 + (w_indices - center_w)**2)
            # we only want to get mean activation position
            # avg_distance = np.sum(distances * grid) / total_activation if total_activation > 0 else 0
            
            # Compute concentration ratio: what percentage of total activation is in the top 25% of active pixels
            #active_values = grid[active_positions]
            #sorted_values = np.sort(active_values)[::-1]  # Sort in descending order
            #quarter_point = max(1, len(sorted_values) // 4)
           # concentration_ratio = np.sum(sorted_values[:quarter_point]) / total_activation if total_activation > 0 else 0
            
            # Compute activation area: percentage of image area that has activations
            #activation_area = len(active_positions[0]) / (HEIGHT * WIDTH)

            # Store metrics
            results[idx] = {
                "feature_id": feature_id,
                "image_id": idx,
                #"spatial_spread": float(avg_distance),
                #"concentration_ratio": float(concentration_ratio),
                #"activation_area": float(activation_area),
                "max_activation": float(grid.max()),
                "total_activation": float(total_activation),
                "center": (float(center_h), float(center_w))
            }
            # print(results[idx])

        # Calculate weighted averages based on total activation in each image
        total_activations = {idx: np.sum(data["grid"]) for idx, data in metrics_by_image.items() if idx in results}
        total_weight = sum(total_activations.values())
        # Weighted center calculation
        weighted_center_h = sum(results[idx]["center"][0] * total_activations[idx] for idx in results) / total_weight if total_weight > 0 else 0
        weighted_center_w = sum(results[idx]["center"][1] * total_activations[idx] for idx in results) / total_weight if total_weight > 0 else 0
        
        # Standard deviation from mean activation position
        # Doing this for each dimension separately because I forgot how to consider them together
        std_dev_h = np.sqrt(sum((results[idx]["center"][0] - weighted_center_h)**2 * total_activations[idx] for idx in results) / total_weight) if total_weight > 0 else 0
        std_dev_w = np.sqrt(sum((results[idx]["center"][1] - weighted_center_w)**2 * total_activations[idx] for idx in results) / total_weight) if total_weight > 0 else 0
        
        # Weighted spatial spread calculation
        #weighted_spread = sum(results[idx]["spatial_spread"] * total_activations[idx] for idx in results) / total_weight if total_weight > 0 else 0
        
        avg_metrics = {
            "feature_id": feature_id,
            #"spatial_spread": float(weighted_spread),
            #"concentration_ratio": float(np.mean([m["concentration_ratio"] for m in results.values()])),
            #"activation_area": float(np.mean([m["activation_area"] for m in results.values()])),
            "num_images": len(results),
            #"index_of_dispersion": float(index_of_dispersion),
            "center": (float(weighted_center_h), float(weighted_center_w)),
            "std_dev": (float(std_dev_h), float(std_dev_w)),
            "entropy": float(entropy),
            "avg_activation_distance": float(avg_distance),
            "activation_pattern": binary_pattern.tolist()  # Include binary pattern in results
        }
        # Add tag for active features
        avg_metrics["active"] = len(results) >= FEATURE_ACTIVATED_IMAGES_THRESHOLD
            
        if avg_metrics["active"]:
            print(avg_metrics)
        return avg_metrics
    
    def compute_spatial_sparsity():
        non_sparse_features = np.zeros(len(scored_storage), dtype=bool)
        img_list = list(image_activations_dir.glob("*.npz"))
        for img in img_list:
            saved = np.load(img)
            ind, wei = saved["arr_0"].ravel(), saved["arr_1"].ravel()
            feature_counts = np.bincount(ind[wei > 0.0], minlength=len(scored_storage))
            non_sparse_features |= feature_counts > FEATURE_ACTIVATED_IMAGES_THRESHOLD
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
        if counts[feature_id] >= FEATURE_ACTIVATED_IMAGES_THRESHOLD:
            metrics = compute_spatial_metrics(feature_id)
            if metrics:
                # Convert numpy values to Python native types for JSON serialization
                metrics_dict = {
                    "center": (float(metrics["center"][0]), float(metrics["center"][1])),
                    "frequency": float(frequencies[feature_id]),
                    "count": int(counts[feature_id]),
                    "max_activation": float(maxima[feature_id]),
                    "entropy": float(metrics["entropy"]),
                    "avg_activation_distance": float(metrics["avg_activation_distance"]),
                    "activation_pattern": metrics["activation_pattern"]
                }
                all_metrics[feature_id] = metrics_dict
    
    # Calculate summary statistics
    #spatial_spreads = np.array([m['spatial_spread'] for m in all_metrics.values()])
    #avg_spread = np.mean(spatial_spreads)
    #var_spread = np.var(spatial_spreads)
    #index_of_dispersion = var_spread / avg_spread if avg_spread > 0 else 0
    
    #avg_concentration = np.mean([m['concentration_ratio'] for m in all_metrics.values()])
    #avg_area = np.mean([m['activation_area'] for m in all_metrics.values()])
    
    # Output summary statistics
    print(f"Total features: {len(scored_storage)}")
    print(f"Features with activations: {len(all_metrics)}")
    
    # Print features with non-zero entropy
    nonzero_entropy_features = {
        fid: metrics for fid, metrics in all_metrics.items() 
        if metrics["entropy"] > 0
    }
    print(f"\nFeatures with non-zero entropy: {len(nonzero_entropy_features)}")
    for fid, metrics in nonzero_entropy_features.items():
        print(f"Feature {fid}: entropy = {metrics['entropy']:.3f}")

    # Prepare JSON output
    metrics_json = {
        "summary": {
            "total_features": len(scored_storage),
            "features_with_activations": len(all_metrics),
            "spatial_sparsity": float(sparsity),
            #"avg_spatial_spread": float(avg_spread),
            #"index_of_dispersion": float(index_of_dispersion),
            #"avg_concentration_ratio": float(avg_concentration),
            #"avg_activation_area": float(avg_area)
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
