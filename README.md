# vaov

## Setup
```
uv sync
mkdir -p somewhere
cd scored-storage; maturin develop; cd ..
git lfs clone https://huggingface.co/nev/taef1 somewhere/taef1
uv run python -m src.vaov.server
uv run python -m src.vaov.sae_trainer
uv run python main.py
```

## Credits
 * https://github.com/davisyoshida/qax -- quantization library for quantized T5 support.
    * [Author](https://github.com/davisyoshida): offered a lot of help and advice for quantization with jax and `qax`. Updated the library twice to support the needs of this project.
 * https://github.com/aredden -- gave advice for Flux accuracy issues
 * https://github.com/markian-rybchuk -- tested the code, suggested improvements, helped with debugging
