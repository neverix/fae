from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm
import os
import argparse
from pathlib import Path


def download_subdirectory(repo_id, subdir, local_dir, repo_type="model"):
    print(f"Downloading {subdir} from {repo_id} to {local_dir}")
    # Initialize HfApi
    api = HfApi()
    
    # List all files in the repository
    print(f"Listing files in {repo_id}")
    all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    
    # Filter files that belong to the target subdirectory
    target_files = [f for f in all_files if f.startswith(f"{subdir}/")]
    
    print(f"Found {len(target_files)} files in {subdir}")
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Download each file individually
    for file in tqdm(target_files):
        print(f"Downloading {file}")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type=repo_type,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Save actual files instead of symlinks
        )
    print(f"Downloaded {len(target_files)} files to {local_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download directories from Hugging Face")
    parser.add_argument("--repo", type=str, default="dmitriihook/flux1-saes",
                        help="HuggingFace repository ID")
    parser.add_argument("--dirs", type=str, nargs="+", 
                        default=["maxacts_itda_50k_256/itda_new_data", "maxacts_itda_50k_256"],
                        help="Directories to download")
    parser.add_argument("--output", type=str, default="somewhere",
                        help="Local directory to save files")
    
    args = parser.parse_args()
    
    for directory in args.dirs:
        # Create a proper local directory path that matches the structure
        local_path = Path(args.output) / directory
        local_path.parent.mkdir(parents=True, exist_ok=True)
        download_subdirectory(
            repo_id=args.repo,
            subdir=directory,
            local_dir=args.output
        )

if __name__ == "__main__":
    main()
