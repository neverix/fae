import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jaxtyping import Array, Float, UInt
from typing import Literal, Optional
import numpy as np
from functools import partial
from collections import defaultdict
import os
import heapq
import json
import shutil
import numba as nb
from loguru import logger
from pathlib import Path
from .scored_storage import ScoredStorage
import math


nf4 = np.asarray(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)


@dataclass(frozen=True)
class SAEConfig:
    d_model: int = 3072
    n_features: int = 32768

    do_update: bool = True

    param_dtype: jax.typing.DTypeLike = jnp.float32
    bias_dtype: jax.typing.DTypeLike = jnp.float32
    clip_data: Optional[float] = 16.0

    k: int = 128
    aux_k: int = 512
    aux_k_coeff: float = 0.125
    dead_after_tokens: int = 200_000
    death_threshold: float = 1.0

    @property
    def dead_after(self):
        return self.dead_after_tokens // self.full_batch_size

    batch_size: int = 4
    seq_len: int = 512 + 256
    seq_mode: Literal["both", "txt", "img"] = "both"
    n_steps: int = 100_000
    wandb_name: Optional[tuple[str, str]] = ("neverix", "vaov")

    tp_size: int = jax.local_device_count()

    learning_rate: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-10
    ema: float = 0.995
    grad_clip_threshold: float = 0.85
    warmup_steps: int = 50

    top_k_activations: int = 1024
    image_max: float = 5.0

    @property
    def real_seq_len(self):
        if self.seq_mode == "txt":
            return 512
        elif self.seq_mode == "img":
            return self.seq_len - 512
        return self.seq_len

    @property
    def full_batch_size(self):
        return self.batch_size * self.real_seq_len

    @property
    def width_and_height(self):
        seq_len = self.seq_len
        width_height_product = (seq_len - 512) * (16 * 16)
        width_and_height = math.isqrt(width_height_product)
        return width_and_height, width_and_height

    def cut_up(
        self, training_data: Float[Array, "*batch seq_len d_model"]
    ) -> Float[Array, "full_batch_size d_model"]:
        training_data = training_data.astype(jnp.bfloat16)
        if self.seq_mode == "txt":
            training_data = training_data[..., :512, :]
        elif self.seq_mode == "img":
            training_data = training_data[..., 512:, :]

        training_data = training_data.reshape(-1, training_data.shape[-1])
        assert training_data.shape == (self.full_batch_size, self.d_model)
        return training_data

    def uncut(
        self, tensor: Float[Array, "full_batch_size *dims"]
    ) -> tuple[Float[Array, "*batch txt_seq_len d_model"], Float[Array, "*batch img_seq_len d_model"]]:
        unbatched = tensor.reshape(self.batch_size, -1, *tensor.shape[1:])
        if self.seq_mode == "txt":
            assert unbatched.shape[1] == 512
            txt = unbatched
            img = unbatched[:, :0]
        elif self.seq_mode == "img":
            assert unbatched.shape[1] == self.seq_len - 512
            img = unbatched
            txt = unbatched[:, :0]
        elif self.seq_mode == "both":
            txt = unbatched[:, :512]
            img = unbatched[:, 512:]
        else:
            raise NotImplementedError()
        return txt, img


class SAEOutputSaver(object):
    def __init__(self, config: SAEConfig, save_dir: os.PathLike):
        self.config = config
        save_dir = Path(save_dir).resolve()
        if save_dir.exists():
            logger.warning(f"Deleting {save_dir} for saving")
            shutil.rmtree(save_dir)
        special_dirs = save_dir / "images", save_dir / "texts", save_dir / "activations", save_dir / "image_activations"
        for dir in special_dirs:
            dir.mkdir(parents=True, exist_ok=True)
        self.images_dir, self.texts_dir, self.activations_dir, self.image_activations_dir = special_dirs
        self.feature_acts = ScoredStorage(
            save_dir / "feature_acts.db",
            4, config.top_k_activations
        )

    def save(
        self,
        sae_weights_txt: Float[Array, "batch txt_seq_len k"],
        sae_weights_img: Float[Array, "batch img_seq_len k"],
        sae_indices_txt: UInt[Array, "batch txt_seq_len k"],
        sae_indices_img: UInt[Array, "batch img_seq_len k"],
        prompts: list[str], images: np.ndarray,
        step: int
    ):
        batch_size = sae_weights_txt.shape[0]
        assert sae_weights_txt.shape[0] == sae_weights_img.shape[0] == sae_indices_txt.shape[0] == sae_indices_img.shape[0]
        assert batch_size == len(prompts) == images.shape[0]
        assert batch_size == self.config.batch_size
        img_seq_len = sae_weights_img.shape[1]
        k = sae_weights_img.shape[2]
        assert k == self.config.k
        assert k == sae_indices_img.shape[2] == sae_weights_txt.shape[2] == sae_indices_txt.shape[2]
        use_img = img_seq_len > 0
        if use_img:
            assert img_seq_len == sae_indices_img.shape[1] == images.shape[-1] * images.shape[-2] // 4
            images_to_save = (images / self.config.image_max).clip(-1, 1)
            images_to_save = np.abs(images_to_save[..., None] - nf4).argmin(-1).astype(np.uint8)
            images_to_save = (
                (images_to_save[..., ::2] & 0x0F)
                | ((images_to_save[..., 1::2] << 4) & 0xF0))
            np.savez(
                self.images_dir / f"{step}.npz",
                images_to_save,
            )
        width = images.shape[-1] // 2
        nums, indices, activations = make_feat_data(sae_indices_img, sae_weights_img, width, step, batch_size, img_seq_len, k, use_img)
        self.feature_acts.insert_many(nums, indices, activations)
        rows, _scores, mask = self.feature_acts.all_rows()
        used_rows = rows[:, :-2][mask].astype(np.uint64)
        unique_idces = np.unique(used_rows[:, 0] * batch_size + used_rows[:, 1])
        unique_rows = np.stack((unique_idces // batch_size, unique_idces % batch_size), axis=1)
        extant_images = set(tuple(map(int, r)) for r in unique_rows)
        self.image_activations_dir.mkdir(parents=True, exist_ok=True)
        for image in self.image_activations_dir.glob("*.npz"):
            identifier = tuple(map(int, image.stem.split("_")))
            if identifier not in extant_images:
                image.unlink()
        for i in range(batch_size):
            identifier = step, i
            if identifier not in extant_images:
                continue
            np.savez(
                self.image_activations_dir / f"{'_'.join(map(str, identifier))}.npz",
                sae_indices_img[i],
                sae_weights_img[i],
            )


@nb.jit
def make_feat_data(sae_indices_img, sae_weights_img, width, step, batch_size, img_seq_len, k, use_img):
    total_activations = sae_indices_img.size
    nums = np.empty(total_activations, dtype=np.uint32)
    indices = np.empty((total_activations, 4), dtype=np.uint32)
    activations = np.empty(total_activations, dtype=np.float32)
    index = 0
    for i in range(batch_size):
        if use_img:
            for x in range(img_seq_len):
                for a in range(k):
                    feature_num, activation = int(sae_indices_img[i, x, a]), float(sae_weights_img[i, x, a])
                    h, w = x // width, x % width
                    # new_data.append((feature_num, (step, i, h, w), activation))
                    nums[index] = feature_num
                    indices[index] = (step, i, h, w)
                    activations[index] = activation
                    index += 1
    return nums, indices, activations
