# vaov

## Setup
```
uv sync
mkdir -p somewhere
wget -c 'https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors' -O somewhere/flux_all.st
uv run python -m src.vaov.refluxer
uv run python -m src.vaov.diformer
JAX_PLATFORMS=cpu uv run python -m src.vaov.diformer --use_torch
```

## Credits
 * https://github.com/davisyoshida/qax -- quantization library for quantized T5 support