from safetensors import safe_open
from safetensors.torch import save_file

if __name__ == "__main__":
    useful = {}
    with safe_open("somewhere/flux_all.st", framework="torch") as f:
        for key in f.keys():
            if not key.startswith("model"):
                continue
            useful[key] = f.get_tensor(key)
    save_file(useful, "somewhere/flux.st")