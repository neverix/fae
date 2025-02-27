from huggingface_hub import HfApi
from glob import glob
import shutil
import os

global_here_path = os.path.abspath(".") + "/"

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

api = HfApi()


shutil.rmtree("somewhere/symlinks", ignore_errors=True)
os.makedirs("somewhere/symlinks", exist_ok=True)
patterns = "maxacts*", "sae_*"
for pattern in patterns:
    for path in glob(f"somewhere/{pattern}"):
        if "maxacts" in pattern:
            os.makedirs(global_here_path + f"somewhere/symlinks/{os.path.basename(path)}", exist_ok=True)
            for filename in ("feature_acts.db.npy", "feature_acts.db.npy.bak", "images", "image_activations"):
                if os.path.isdir(path + "/" + filename):
                    if not os.path.exists(path + "/" + filename + ".zip"):
                        shutil.make_archive(path + "/" + filename, 'zip', path + "/" + filename)
                    filename = filename + ".zip"
                fpath = path + "/" + filename
                os.symlink(global_here_path + fpath, global_here_path + f"somewhere/symlinks/{os.path.basename(path)}/{filename}", target_is_directory=True)
        else:
            os.symlink(global_here_path + path, global_here_path + f"somewhere/symlinks/{os.path.basename(path)}", target_is_directory=True)
        
    
api.upload_large_folder(
    folder_path="somewhere/symlinks",
    repo_id="nev/flux1-saes",
    repo_type="model",
)