# vaov

## Setup
```
uv sync
mkdir -p somewhere
uv run python -m src.vaov.server
```

## Credits
 * https://github.com/davisyoshida/qax -- quantization library for quantized T5 support.
    * Author: offered a lot of help and advice for quantization with jax and `qax`.
 * https://github.com/aredden -- gave advice for Flux accuracy issues
 * https://github.com/markian-rybchuk -- tested the code, suggested improvements, helped with debugging