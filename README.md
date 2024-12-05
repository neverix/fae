# fae

Flux implementation with SAEs

## Setup
```
uv sync
mkdir -p somewhere
git lfs clone https://huggingface.co/nev/taef1 somewhere/taef1
uv run python -m src.fae.server
```

## Credits
 * https://github.com/davisyoshida/qax -- quantization library for quantized T5 support.
    * [Author](https://github.com/davisyoshida): offered a lot of help and advice for quantization with jax and `qax`. Updated the library twice to support the needs of this project.
 * https://github.com/aredden -- gave advice for Flux accuracy issues
 * https://github.com/markian-rybchuk -- tested the code, suggested improvements, helped with debugging
