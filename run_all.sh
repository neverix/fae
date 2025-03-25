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
# Run with default settings
# uv run python run_fae.py

# Specify a different cache path
# uv run python run_fae.py --cache-path="somewhere/other_cache_dir"

# Change image dimensions
# uv run python run_fae.py --width=768 --height=768

# Use a different port
# uv run python run_fae.py --port=5002

# Only compute and output metrics without starting the server
# uv run python run_fae.py --metrics-only

# Clear the image cache before starting
# uv run python run_fae.py --clear-image-cache

# uv run python -m src.fae.sae_trainer --train_mode=False -seq_len=1536 --batch_size=4 --restore_from=somewhere/sae_double_l18_img
# uv run python -m src.fae.sae_trainer --train_mode=True -seq_len=1536 --timesteps=4 --batch_size=4 --restore_from=somewhere/sae_double_l18_img
# uv run python -m src.fae.sae_trainer
# uv run python -m src.fae.sae_trainer --train_mode=False

# uv run python -m src.fae.sae_trainer --layer=18 --block_type=double --train_mode=False --restore_from=somewhere/sae_double_l18_img
# uv run python -m src.fae.sae_trainer --layer=12 --block_type=double --sae_train_every=1

# uv run python -m src.fae.sae_trainer --layer=3 --block_type=double
# uv run python -m src.fae.sae_trainer --layer=3 --block_type=double --train_mode=False
# uv run python -m src.fae.sae_trainer --layer=6 --block_type=double
# uv run python -m src.fae.sae_trainer --layer=6 --block_type=double --train_mode=False
uv run python -m src.fae.sae_trainer --layer=15 --block_type=double
uv run python -m src.fae.sae_trainer --layer=15 --block_type=double --train_mode=False