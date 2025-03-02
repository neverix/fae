#!/bin/bash

set -e
# uv run python -m src.fae.sae_trainer -k=32
# uv run python -m src.fae.sae_trainer -k=32 --train_mode=False
# uv run python -m src.fae.sae_trainer --layer=18 --block_type="single"
# uv run python -m src.fae.sae_trainer -k=48

# uv run python -m src.fae.sae_trainer --layer=18 --block_type="single" --train_mode=False
# uv run python -m src.fae.sae_trainer --layer=9 --train_mode=False
# uv run python -m src.fae.sae_trainer -k=48
# uv run python -m src.fae.sae_trainer -k=48 --train_mode=False

# uv run python -m src.fae.sae_trainer --layer=9 --block_type="single"
# uv run python -m src.fae.sae_trainer --layer=9 --block_type="single" --train_mode=False

# 768 = 512 + (256 = 16*16 = 256/16 * 256/16)
# 1536 = 512 + (1024 = 32*32 = 512/16 * 512/16)

uv run python -m src.fae.sae_trainer --train_mode=False -seq_len=1536 --batch_size=4 --restore_from=somewhere/sae_double_l18_img