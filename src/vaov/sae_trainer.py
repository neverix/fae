from collections import defaultdict
from fire import Fire
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from datasets import load_dataset
from more_itertools import chunked
from .ensemble import FluxEnsemble
from dataclasses import dataclass, replace
from functools import partial
from jaxtyping import Float, Array, UInt
from loguru import logger
from tqdm.auto import tqdm
from optax._src.linear_algebra import global_norm
import optax
import heapq
import queue
import jax.numpy as jnp
import equinox as eqx
import wandb
import math
import threading
import shutil
import jax
import sys
import os
import json
from pathlib import Path
import orbax.checkpoint as ocp
from typing import Sequence, Literal
from concurrent.futures import ThreadPoolExecutor
import asyncio
from .sae_common import SAEConfig, SAEOutputSaver


class SAEConfigHandler(ocp.type_handlers.TypeHandler):
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)

    def typestr(self):
        return "SAEConfig"

    async def metadata(self, infos: Sequence[ocp.type_handlers.ParamInfo]):
        return [ocp.metadata.Metadata(name=info.name, directory=info.path) for info in infos]

    async def serialize(self, values, infos, args = None):
        futures = []
        for value, info in zip(values, infos):
            info.path.mkdir(exist_ok=True)
            futures.append(
                self._executor.submit(
                    lambda: json.dump(value.__dict__, open(info.path / "config.json", "w"))
                )
            )
        return futures

    async def deserialize(self, infos, args = None):
        del args
        futures = []
        for info in infos:
            futures.append(
                self._executor.submit(
                    lambda: SAEConfig(**json.load(open(info.path / "config.json")))
                )
            )
        return await asyncio.gather(*futures)


ocp.type_handlers.register_type_handler(SAEConfig, SAEConfigHandler(), override=True)


class SAEOutput(eqx.Module):
    x_normed: Float[Array, "batch_size d_model"]
    x: Float[Array, "batch_size d_model"]
    k_weights: Float[Array, "batch_size k"]
    k_indices: UInt[Array, "batch_size k"]
    y_normed: Float[Array, "batch_size d_model"]
    y: Float[Array, "batch_size d_model"]
    losses: dict[str, Float[Array, "batch_size"]]
    loss: Float[Array, "batch_size"]
    fvu: Float[Array, "batch_size"]
    var_explained: Float[Array, ""]


class SAEInfo(eqx.Module):
    config: SAEConfig = eqx.field(static=True)
    n_steps: UInt[Array, ""]
    avg_norm: Float[Array, ""]
    feature_density: Float[Array, "n_features"]
    activated_in: UInt[Array, "batch_size n_features"]
    weight_norms: dict
    weight_grad_norms: dict
    grad_clip_percent: Float[Array, ""]

    @classmethod
    def create(cls, config: SAEConfig, W_enc: Float[Array, "d_model n_features"]) -> "SAEInfo":
        return cls(
            config=config,
            n_steps=jnp.zeros(()),
            avg_norm=jnp.ones(()),
            feature_density=jnp.zeros((config.n_features,)),
            activated_in=jnp.zeros((config.n_features,), dtype=jnp.uint32),
            grad_clip_percent=jnp.zeros(()),
            weight_norms=dict(W_enc=jnp.linalg.norm(W_enc), W_dec=jnp.linalg.norm(W_enc.T)),
            weight_grad_norms=dict(W_enc=jnp.zeros(()), W_dec=jnp.zeros(()))
        )

    @property
    def tgt_norm(self):
        return math.sqrt(self.config.d_model)

    def norm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        return x / self.avg_norm * self.tgt_norm

    def denorm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        return x / self.tgt_norm * self.avg_norm

    def step(self, sae: "SAE", grads: "SAE", updates: "SAE", outputs: SAEOutput):
        weighting_factor, new_weighting_factor = self.n_steps / (self.n_steps + 1), 1 / (self.n_steps + 1)

        if self.config.do_update:
            new_avg_norm = jnp.mean(jnp.linalg.norm(outputs.x, axis=-1))
            updated_avg_norm = self.avg_norm * weighting_factor + new_avg_norm * new_weighting_factor
        else:
            updated_avg_norm = self.avg_norm

        activations = jnp.zeros(self.feature_density.shape).at[outputs.k_indices.flatten()].add(1)
        new_feature_density = activations / self.config.full_batch_size
        updated_feature_density = self.feature_density * weighting_factor + new_feature_density * new_weighting_factor

        updated_activated_in = jnp.where(activations > 0, 0, self.activated_in + 1)

        if self.config.do_update:
            # https://github.com/google-deepmind/optax/blob/63cdeb4ada95498626a52d209a029210ac066aa1/optax/transforms/_clipping.py#L91
            new_grad_clip_percent = (
                (global_norm(grads) > self.config.grad_clip_threshold)
                .squeeze()
                .astype(jnp.float32)
            )
            updated_grad_clip_percent = self.grad_clip_percent * weighting_factor + new_grad_clip_percent * new_weighting_factor
        else:
            updated_grad_clip_percent = self.grad_clip_percent

        new_weight_norms = dict(W_enc=jnp.linalg.norm(sae.W_enc), W_dec=jnp.linalg.norm(sae.W_dec))
        new_weight_grad_norms = dict(
            W_enc=jnp.linalg.norm(grads.W_enc), W_dec=jnp.linalg.norm(grads.W_dec)
        )

        return replace(self,
            n_steps=self.n_steps + 1,
            avg_norm=updated_avg_norm,
            feature_density=updated_feature_density,
            activated_in=updated_activated_in,
            grad_clip_percent=updated_grad_clip_percent,
            weight_norms=new_weight_norms,
            weight_grad_norms=new_weight_grad_norms
        )

    @classmethod
    def pspec(cls, config: SAEConfig) -> "SAEInfo":
        return SAEInfo(
            config=config,
            n_steps=P(),
            avg_norm=P(),
            feature_density=P("tp"),
            activated_in=P("tp"),
            grad_clip_percent=P(),
            weight_norms=dict(W_enc=P(), W_dec=P()),
            weight_grad_norms=dict(W_enc=P(), W_dec=P())
        )


SPARSE_MATMUL_MAX_BATCH = 32768
@jax.remat
def sparse_matmul_scan(
    weights: Float[Array, "batch_size k"],
    indices: UInt[Array, "batch_size k"],
    W: Float[Array, "n_features d_model"]) -> Float[Array, "batch_size d_model"]:
    def sparse_matmul_basic(wi):
        weights, indices = wi
        return (weights[:, None] * W[indices]).sum(0)
    return jax.lax.map(sparse_matmul_basic, (weights, indices),
                       batch_size=SPARSE_MATMUL_MAX_BATCH // weights.shape[-1])


def sparse_matmul(
    weights: Float[Array, "batch_size k"],
    indices: UInt[Array, "batch_size k"],
    W: Float[Array, "n_features d_model"]) -> Float[Array, "batch_size d_model"]:
    if weights.shape != indices.shape:
        raise ValueError("Weights and indices must have the same shape")
    if weights.ndim > 2:
        return jax.vmap(sparse_matmul, in_axes=(0, 0, None), out_axes=0)(weights, indices, W)
    return sparse_matmul_scan(weights, indices, W)

class SAE(eqx.Module):
    config: SAEConfig = eqx.field(static=True)
    info: SAEInfo

    b_pre: Float[Array, "d_model"]
    W_enc: Float[Array, "d_model n_features"]
    b_mid: Float[Array, "n_features"]
    W_dec: Float[Array, "n_features d_model"]
    b_post: Float[Array, "n_features"]

    @classmethod
    def create(cls, config: SAEConfig, key: jax.random.PRNGKey):
        W_enc = jax.nn.initializers.orthogonal()(
            key, (config.d_model, config.n_features), dtype=config.param_dtype
        )
        W_dec = W_enc.T
        W_dec = W_dec / jnp.linalg.norm(W_dec, axis=-1, keepdims=True)
        return cls(
            config=config,
            b_pre=jnp.zeros((config.d_model,), dtype=config.bias_dtype),
            W_enc=W_enc,
            b_mid=jnp.zeros((config.n_features,), dtype=config.bias_dtype),
            W_dec=W_dec,
            b_post=jnp.zeros((config.d_model,), dtype=config.bias_dtype),
            info=SAEInfo.create(config, W_enc)
        )

    def __call__(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        info = jax.lax.stop_gradient(self.info)
        x_normed = info.norm(x)
        x_normed = jnp.clip(
            x_normed,
            -self.config.clip_data,
            self.config.clip_data,
        )
        encodings = (x_normed - self.b_post - self.b_pre) @ self.W_enc
        weights, indices = jax.lax.approx_max_k(encodings, self.config.k)
        decoded = sparse_matmul(weights, indices, self.W_dec)
        y_normed = decoded + self.b_post
        recon_loss = jnp.mean(jnp.square(x_normed - y_normed), axis=-1)
        y = info.denorm(y_normed)

        dead_activated_in, dead_indices = jax.lax.top_k(info.activated_in, self.config.aux_k)
        dead_weights = encodings[..., dead_indices]
        dead_weights = jnp.where(
            dead_activated_in > self.config.dead_after, dead_weights, 0
        )

        dead_indices = jnp.broadcast_to(dead_indices, dead_weights.shape)
        aux_y_normed = sparse_matmul(dead_weights, dead_indices, self.W_dec) + self.b_post
        aux_k_loss = jnp.mean(jnp.square(x_normed - aux_y_normed), axis=-1)
        aux_k_loss = jnp.where(info.n_steps >= self.config.dead_after, aux_k_loss, 0)

        loss = recon_loss + self.config.aux_k_coeff * aux_k_loss

        fvu = jnp.mean(jnp.square(x_normed - y_normed)) / jnp.mean(jnp.square(x_normed))
        correlation = ((x_normed - x_normed.mean(axis=0)) * (y_normed - y_normed.mean(axis=0))).mean(axis=0)
        var_explained = (jnp.square(correlation) / (
            jnp.var(x_normed, axis=0) * jnp.var(y_normed, axis=0)
        )).mean()

        return SAEOutput(
            x_normed=x_normed,
            x=x,
            k_weights=weights,
            k_indices=indices,
            y_normed=y_normed,
            y=y,
            losses=dict(
                recon_loss=recon_loss,
                aux_k_loss=aux_k_loss
            ),
            loss=loss,
            fvu=fvu,
            var_explained=var_explained
        )

    def process_gradients(self, grads: "SAE") -> "SAE":
        W_dec = grads.W_dec
        W_dec = W_dec - jnp.einsum("ij,ij->i", W_dec, self.W_dec)[:, None] * self.W_dec
        # W_dec @ self.W_dec.T = 0
        return replace(grads, W_dec=W_dec)

    def apply_updates(self, updates: "SAE", grads: "SAE", past_output: SAEOutput) -> "SAE":
        if not self.config.do_update:
            self = eqx.tree_at(lambda x: x.info, self,
                                replace_fn=lambda info: info.step(self, updates, grads, past_output))
            return self
        updated = eqx.apply_updates(self, updates)
        updated = eqx.tree_at(
            lambda x: x.b_post,
            updated,
            replace_fn=lambda b_post: jnp.where(
                self.info.n_steps > 1, b_post, past_output.x_normed.mean(axis=0)
            ),
        )
        updated = eqx.tree_at(lambda x: x.W_dec, updated,
                              replace_fn=lambda W_dec: W_dec / jnp.linalg.norm(W_dec, axis=-1, keepdims=True))
        updated = eqx.tree_at(lambda x: x.info, updated,
                              replace_fn=lambda info: info.step(updated, updates, grads, past_output))
        return updated

    def split(self):
        return eqx.partition(self,
                             lambda x: eqx.is_array(x) and not isinstance(x, (SAEInfo, SAEConfig)),
                             is_leaf=lambda x: isinstance(x, (SAEInfo, SAEConfig)))

    @classmethod
    def pspec(cls, config: SAEConfig) -> "SAE":
        return cls(
            config=config,
            b_pre=P(None),
            W_enc=P(None, "tp"),
            b_mid=P("tp"),
            W_dec=P("tp", None),
            b_post=P(None),
            info=SAEInfo.pspec(config),
        )

def hist(x):
    return wandb.Histogram(np.asarray(x.flatten().astype(jnp.float32).tolist()))

class SAETrainer(eqx.Module):
    config: SAEConfig
    sae_params: SAE
    sae_logic: SAE
    ema: SAE
    optimizer: optax.GradientTransformation
    optimizer_state: SAE

    @classmethod
    def create(cls, config: SAEConfig, key: jax.random.PRNGKey):
        logger.info("Creating mesh")
        shape_request = (-1, config.tp_size)
        device_count = jax.device_count()
        mesh_shape = np.arange(device_count).reshape(*shape_request).shape
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
        mesh = jax.sharding.Mesh(physical_mesh, ("dp", "tp"))

        def sae_creator(key):
            return SAE.create(config, key)

        logger.info("Creating SAE")
        pspec = SAE.pspec(config)
        sae = jax.jit(
            sae_creator,
            out_shardings=jax.tree.map(
                lambda x: jax.sharding.NamedSharding(mesh, x), pspec
            ),
        )(key)
        sae_params, sae_logic = sae.split()

        logger.info("Creating optimizer")
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps, decay_steps=config.n_steps)
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_threshold),
            optax.adam(scheduler, b1=config.beta1, b2=config.beta2, eps=config.eps)
        )
        optimizer_state = optimizer.init(sae_params)

        logger.info("Creating EMA")
        ema = jax.tree.map(jnp.copy, sae_params)

        return cls(
            config=config,
            sae_params=sae_params,
            sae_logic=sae_logic,
            ema=ema,
            optimizer=optimizer,
            optimizer_state=optimizer_state
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    @partial(jax.grad, has_aux=True)
    def loss_fn(sae_params: SAE, sae_logic: SAE,
                x: Float[Array, "batch_size d_model"]) -> tuple[Float[Array, ""], SAEOutput]:
        sae = eqx.combine(sae_params, sae_logic)
        outputs = sae(x)
        return outputs.loss.mean(), outputs

    @partial(eqx.filter_jit, donate="all")
    def step(self, x: Float[Array, "batch_size d_model"]) -> tuple["SAETrainer", SAEOutput]:
        sae_grad, sae_outputs = self.loss_fn(self.sae_params, self.sae_logic, x)

        sae = eqx.combine(self.sae_params, self.sae_logic)
        if not self.sae_logic.config.do_update:
            sae_grad = jax.tree.map(jnp.zeros_like, sae_grad)
        sae_grad = sae.process_gradients(sae_grad)
        updates, optimizer_state = self.optimizer.update(sae_grad, self.optimizer_state, self.sae_params)
        start_updating = self.sae_logic.config.do_update & (self.sae_logic.info.n_steps > 1)
        updates = jax.tree.map(
            lambda u: jnp.where(start_updating, u, 0),
            updates)
        optimizer_state = jax.tree.map(
            lambda o, ld: jnp.where(start_updating, o, ld),
            optimizer_state, self.optimizer_state
        )

        updated_sae = sae.apply_updates(updates, sae_grad, sae_outputs)
        sae_params, sae_logic = updated_sae.split()
        updated_ema = jax.tree.map(
            lambda ema, sae: jnp.where(start_updating,
                                       ema * self.config.ema + sae * (1 - self.config.ema), 0),
            self.ema, sae_params)
        return replace(self, sae_params=sae_params, sae_logic=sae_logic,
                       ema=updated_ema, optimizer_state=optimizer_state), sae_outputs

    def log_dict(self, sae_outputs: SAEOutput) -> dict[str, (float | wandb.Histogram)]:
        log_dict = {
            "recon_loss": sae_outputs.losses["recon_loss"].mean(),
            "aux_k_loss": sae_outputs.losses["aux_k_loss"].mean(),
            "loss": sae_outputs.loss.mean(),
            "grad_clip_percent": self.sae_logic.info.grad_clip_percent,
            "feature_density": hist(self.sae_logic.info.feature_density),
            "activated_in": hist(self.sae_logic.info.activated_in),
            "data_norm": self.sae_logic.info.avg_norm,
            "fvu": sae_outputs.fvu.mean(),
            "var_explained": sae_outputs.var_explained,
            "W_enc_norm": self.sae_logic.info.weight_norms["W_enc"],
            "W_dec_norm": self.sae_logic.info.weight_norms["W_dec"],
            "W_enc_grad_norm": self.sae_logic.info.weight_grad_norms["W_enc"],
            "W_dec_grad_norm": self.sae_logic.info.weight_grad_norms["W_dec"],
            "tokens_processed": self.sae_logic.info.n_steps * self.config.full_batch_size
        }
        return {
            k: (float(v) if not isinstance(v, wandb.Histogram) else v)
            for k, v in log_dict.items()
        }


class SAEOverseer:
    def __init__(self, config: SAEConfig, save_at="somewhere/sae_mid", save_every=1000, restore=False):
        self.config = config
        self.key = jax.random.PRNGKey(0)
        self.sae_trainer = SAETrainer.create(config, self.key)

        if config.wandb_name:
            self.run = wandb.init(
                entity=config.wandb_name[0], project=config.wandb_name[1], config=config
            )
        self.bar = tqdm(total=config.n_steps)

        self.save_at = None
        if save_at:
            save_at = Path(save_at).resolve()
            if not restore:
                save_at = ocp.test_utils.erase_and_create_empty(save_at)
            self.save_at = save_at
            options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=save_every)
            self.save_every = save_every
            self.mngr = ocp.CheckpointManager(
                save_at,
                ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                options=options,
            )
            if restore:
                self.restore()

    def step(self, x: Float[Array, "batch_size d_model"]) -> SAEOutput:
        self.sae_trainer, sae_outputs = self.sae_trainer.step(x)
        log_dict = self.sae_trainer.log_dict(sae_outputs)
        pure_log_dict = {k: v for k, v in log_dict.items() if not isinstance(v, wandb.Histogram)}
        histograms = {k: v for k, v in log_dict.items() if isinstance(v, wandb.Histogram)}
        step = int(self.sae_trainer.sae_logic.info.n_steps)
        self.run.log(pure_log_dict, step=step)
        for k, v in histograms.items():
            self.run.log({k: v}, step=step)
        self.bar.update()
        self.bar.set_postfix(pure_log_dict)
        if self.save_at and self.save_every:
            self.save(step)
        return sae_outputs

    def save(self, step: int):
        save_dict = dict(
            sae_params=self.sae_trainer.sae_params,
            info=self.sae_trainer.sae_logic.info,
            ema=self.sae_trainer.ema,
            optimizer_state=self.sae_trainer.optimizer_state
        )
        self.mngr.save(step, save_dict)


    def restore(self):
        save_dict = self.mngr.restore(
            self.mngr.latest_step(),
            items=dict(
                sae_params=self.sae_trainer.sae_params,
                info=self.sae_trainer.sae_logic.info,
                ema=self.sae_trainer.ema,
                optimizer_state=self.sae_trainer.optimizer_state
            ),
        )
        self.sae_trainer = replace(self.sae_trainer,
                                    sae_params=save_dict["sae_params"],
                                    sae_logic=eqx.tree_at(lambda x: x.info, self.sae_trainer.sae_logic,
                                                         save_dict["info"]),
                                    ema=save_dict["ema"],
                                    optimizer_state=save_dict["optimizer_state"])
        self.bar.n = int(self.sae_trainer.sae_logic.info.n_steps)
        self.bar.last_print_n = int(self.sae_trainer.sae_logic.info.n_steps)


def main(train_mode=False):
    logger.info("Loading dataset")
    prompts_dataset = load_dataset("k-mktr/improved-flux-prompts")
    prompts_iterator = prompts_dataset["train"]["prompt"]
    logger.info("Creating Flux")
    ensemble = FluxEnsemble(use_schnell=True, use_fsdp=True)
    logger.info("Creating SAE trainer")
    config = SAEConfig(
        do_update=train_mode,
        seq_mode="img",
    )
    sae_trainer = SAEOverseer(
        config,
        save_every=None if not train_mode else 1000,
        restore="somewhere/sae_mid" if not train_mode else None,
    )
    if not train_mode:
        saver = SAEOutputSaver(config, Path("somewhere/maxacts"))
    width, height = config.width_and_height
    appeared_prompts = set()
    cycle_detected = False
    for step, prompts in zip(range(config.n_steps), chunked(prompts_iterator, config.batch_size)):
        if len(prompts) < config.batch_size:
            logger.warning("End of dataset")
            continue
        if not cycle_detected:
            new_prompts = set(prompts)
            if new_prompts & appeared_prompts:
                cycle_detected = True
                logger.warning("Cycle detected")
            appeared_prompts |= new_prompts
        key = jax.random.key(step)
        logger.remove()
        images, reaped = ensemble.sample(
            prompts,
            debug_mode=True,
            decode_latents=False,
            sample_steps=1,
            key=key,
            width=width,
            height=height
        )
        assert isinstance(images, jnp.ndarray)  # to silence mypy
        logger.add(sys.stderr, level="INFO")
        training_data = jnp.concatenate((reaped["double_img"][-1], reaped["double_txt"][-1]), axis=-2)
        training_data = config.cut_up(training_data)
        for v in reaped.values():
            v.delete()
        sae_outputs = sae_trainer.step(training_data)
        if not train_mode:
            sae_weights, sae_indices = map(
                config.uncut, map(
                    np.asarray,
                    jax.block_until_ready(
                        jax.device_put(
                            (sae_outputs.k_weights, sae_outputs.k_indices),
                            jax.devices("cpu")[0]
                        )
                    )
                )
            )
            saver.save(*sae_weights, *sae_indices, prompts, np.asarray(images), step)

    print("Waiting for saver to leave...")


if __name__ == "__main__":
    Fire(main)
