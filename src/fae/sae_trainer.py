from fire import Fire
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from datasets import load_dataset
from more_itertools import chunked
from .ensemble import FluxEnsemble
from dataclasses import replace
from functools import partial
from jaxtyping import Float, Array, UInt
from loguru import logger
from tqdm.auto import tqdm
from optax._src.linear_algebra import global_norm
import optax
import jax.numpy as jnp
from .hadamard import hadamard_matrix
import equinox as eqx
import wandb
import math
import jax
import sys
import json
from pathlib import Path
import orbax.checkpoint as ocp
from typing import Sequence
from concurrent.futures import ThreadPoolExecutor
import asyncio
from .sae_common import SAEConfig, SAEOutputSaver
from .interp_globals import post_double_stream, post_single_stream


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
    
    # Normalization/preprocessing
    avg_norm: Float[Array, ""]
    feature_means: Float[Array, "d_model"]  # Per-feature means
    feature_square_means: Float[Array, "d_model"]  # Per-feature expected squared values
    # hadamard or PCA
    whitening_matrix: Float[Array, "d_model d_model"]
    
    # Feature statistics
    feature_density: Float[Array, "n_features"]
    activated_in: UInt[Array, "batch_size n_features"]
    
    # Training statistics
    n_steps: UInt[Array, ""]
    weight_norms: dict
    weight_grad_norms: dict
    grad_clip_percent: Float[Array, ""]
    grad_global_norm: Float[Array, ""]

    @classmethod
    def create(cls, config: SAEConfig, W_enc: Float[Array, "d_model n_features"]) -> "SAEInfo":
        return cls(
            config=config,
            n_steps=jnp.zeros(()),
            
            avg_norm=jnp.ones(()),
            feature_means=jnp.zeros((config.d_model,)),
            feature_square_means=jnp.zeros((config.d_model,)),
            whitening_matrix=(
                None
                if not config.use_whitening else (
                    jnp.eye(config.d_model)
                    if config.use_pca else
                    hadamard_matrix(config.d_model) / jnp.sqrt(self.config.d_model)
                )
            ),

            feature_density=jnp.zeros((config.n_features,)),
            activated_in=jnp.zeros((config.n_features,), dtype=jnp.uint32),

            grad_clip_percent=jnp.zeros(()),
            grad_global_norm=jnp.zeros(()),
            weight_norms=dict(W_enc=jnp.linalg.norm(W_enc), W_dec=jnp.linalg.norm(W_enc.T)),
            weight_grad_norms=dict(W_enc=jnp.zeros(()), W_dec=jnp.zeros(())),
        )

    @property
    def tgt_norm(self):
        return math.sqrt(self.config.d_model)

    @property
    def feature_stds(self):
        return jnp.sqrt(jnp.maximum(self.feature_square_means - jnp.square(self.feature_means), 1e-6))

    def preprocess(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        if self.config.use_whitening:
            return x @ self.whitening_matrix
        return x

    def deprocess(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        if self.config.use_whitening:
            return x @ self.whitening_matrix.T
        return x

    def norm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        if self.config.standardize:
            return (self.preprocess(x) - self.feature_means) / self.feature_stds
        return x / self.avg_norm * self.tgt_norm

    def denorm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        if self.config.standardize:
            return self.deprocess(x * self.feature_stds + self.feature_means)
        return x / self.tgt_norm * self.avg_norm

    def step(self, sae: "SAE", updates: "SAE", grads: "SAE", outputs: SAEOutput):
        # def compute_whitener():
        #     x = outputs.x
        #     x_dec = x - x.mean(axis=0, keepdims=True)
        #     _u, _s, vt = jnp.linalg.svd(x_dec, full_matrices=True)
        #     return vt
        
        # # ugly hack
        # self = replace(
        #     self,
        #     whitening_matrix=jax.lax.cond(
        #         self.n_steps == 0 & self.config.use_pca,
        #         compute_whitener,
        #         lambda: self.whitening_matrix,
        #     ))
        
        weighting_factor, new_weighting_factor = self.n_steps / (self.n_steps + 1), 1 / (self.n_steps + 1)

        if self.config.do_update:
            preprocessed_data = self.preprocess(outputs.x)
            
            new_avg_norm = jnp.mean(jnp.linalg.norm(preprocessed_data, axis=-1))
            updated_avg_norm = self.avg_norm * weighting_factor + new_avg_norm * new_weighting_factor
            
            new_feature_means = jnp.mean(preprocessed_data, axis=0)  # Per-feature mean
            new_feature_square_means = jnp.mean(jnp.square(preprocessed_data), axis=0)    # Per-feature std

            updated_feature_means = self.feature_means * weighting_factor + new_feature_means * new_weighting_factor
            updated_feature_square_means = self.feature_square_means * weighting_factor + new_feature_square_means * new_weighting_factor
        else:
            updated_avg_norm = self.avg_norm
            updated_feature_means = self.feature_means
            updated_feature_square_means = self.feature_square_means

        activations = jnp.zeros(
            self.feature_density.shape, dtype=jnp.uint32
        ).at[outputs.k_indices.flatten()].add(1)
        not_dead = jnp.zeros(
            self.feature_density.shape, dtype=jnp.uint32
        ).at[outputs.k_indices.flatten()].add(outputs.k_weights.flatten() > self.config.death_threshold)
        new_feature_density = activations / self.config.train_batch_size
        # new_feature_density = (activations > 0).astype(jnp.float32)  # so it's easier to see'
        updated_feature_density = self.feature_density * weighting_factor + new_feature_density * new_weighting_factor

        updated_activated_in = jnp.where(not_dead > 0, 0, self.activated_in + 1)

        if self.config.do_update:
            # https://github.com/google-deepmind/optax/blob/63cdeb4ada95498626a52d209a029210ac066aa1/optax/transforms/_clipping.py#L91
            new_grad_global_norm = global_norm(grads)
            new_grad_clip_percent = (
                (new_grad_global_norm > self.config.grad_clip_threshold)
                .squeeze()
                .astype(jnp.float32)
            )
            updated_grad_clip_percent = self.grad_clip_percent * weighting_factor + new_grad_clip_percent * new_weighting_factor
            # updated_grad_global_norm = self.grad_global_norm * weighting_factor + new_grad_global_norm * new_weighting_factor
            updated_grad_global_norm = new_grad_global_norm
        else:
            updated_grad_clip_percent = self.grad_clip_percent
            updated_grad_global_norm = self.grad_global_norm

        new_weight_norms = dict(W_enc=jnp.linalg.norm(sae.W_enc), W_dec=jnp.linalg.norm(sae.W_dec))
        new_weight_grad_norms = dict(
            W_enc=jnp.linalg.norm(grads.W_enc), W_dec=jnp.linalg.norm(grads.W_dec)
        )

        return replace(self,
            n_steps=self.n_steps + 1,
            avg_norm=updated_avg_norm,
            feature_means=updated_feature_means,  # Update per-feature means
            feature_square_means=updated_feature_square_means,  # Update per-feature stds
            feature_density=updated_feature_density,
            activated_in=updated_activated_in,
            grad_global_norm=updated_grad_global_norm,
            grad_clip_percent=updated_grad_clip_percent,
            weight_norms=new_weight_norms,
            weight_grad_norms=new_weight_grad_norms,
        )

    @classmethod
    def pspec(cls, config: SAEConfig) -> "SAEInfo":
        return SAEInfo(
            config=config,
            n_steps=P(),
            avg_norm=P(),
            feature_means=P(None),
            feature_square_means=P(None),
            whitening_matrix=(
                None if not config.use_whitening else P(None)
            ),
            feature_density=P("tp"),
            activated_in=P("tp"),
            grad_clip_percent=P(),
            grad_global_norm=P(),
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
        # mimic kaiming init
        W_enc = W_enc * math.sqrt(config.n_features / config.d_model) / math.sqrt(3)
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
        if self.config.clip_data is not None:
            x_normed = jnp.clip(
                x_normed,
                -self.config.clip_data,
                self.config.clip_data,
            )
        encodings = (x_normed - self.b_post + self.b_pre) @ self.W_enc
        weights, indices = jax.lax.approx_max_k(encodings, self.config.k)
        decoded = sparse_matmul(weights, indices, self.W_dec)
        y_normed = decoded + self.b_post
        recon_loss = jnp.mean(jnp.square(x_normed - y_normed), axis=-1)
        y = info.denorm(y_normed)

        dead_condition = info.activated_in > self.config.dead_after
        dead_weights, dead_indices = jax.lax.approx_max_k(
            jnp.where(dead_condition, encodings, -jnp.inf),
            self.config.aux_k
        )
        dead_weights = jnp.nan_to_num(dead_weights, neginf=0.0)

        aux_y_normed = sparse_matmul(dead_weights, dead_indices, self.W_dec)
        if self.config.aux_k_variant == "openai":
            aux_k_loss = jnp.mean(jnp.square((y_normed - x_normed) - aux_y_normed), axis=-1)
        elif self.config.aux_k_variant == "mine":
            aux_k_loss = jnp.mean(jnp.square(y_normed - (aux_y_normed + self.b_post)), axis=-1)
        else:
            logger.warning("Unknown aux_k_variant", self.config.aux_k_variant)
            aux_k_loss = jnp.zeros_like(recon_loss)
        aux_k_loss = jnp.where(dead_condition.any(), aux_k_loss, 0)

        loss = recon_loss + self.config.aux_k_coeff * aux_k_loss

        fvu = jnp.mean(jnp.square(x_normed - y_normed)) / jnp.mean(jnp.square(x_normed))
        correlation = ((x_normed - x_normed.mean(axis=0)) * (y_normed - y_normed.mean(axis=0))).mean(axis=0)
        var_explained = jnp.nan_to_num(jnp.square(correlation) / (
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
    mesh: jax.sharding.Mesh
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
            warmup_steps=config.warmup_steps, decay_steps=config.n_steps,
            end_value=0.0 if config.decay_lr else config.learning_rate)
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_threshold),
            optax.adam(scheduler, b1=config.beta1, b2=config.beta2, eps=config.eps)
        )
        optimizer_state = optimizer.init(sae_params)

        logger.info("Creating EMA")
        ema = jax.tree.map(jnp.copy, sae_params)

        return cls(
            mesh=mesh,
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
            "grad_global_norm": self.sae_logic.info.grad_global_norm,
            "grad_clip_percent": self.sae_logic.info.grad_clip_percent,
            "feature_density": hist(jnp.log(jnp.clip(self.sae_logic.info.feature_density, 1e-6, 1))),
            "pct_dead": (self.sae_logic.info.activated_in > self.config.dead_after).mean(),
            "activated_in": hist(self.sae_logic.info.activated_in),
            "data_norm": self.sae_logic.info.avg_norm,
            "fvu": sae_outputs.fvu.mean(),
            "var_explained": sae_outputs.var_explained,
            "W_enc_norm": self.sae_logic.info.weight_norms["W_enc"],
            "W_dec_norm": self.sae_logic.info.weight_norms["W_dec"],
            "W_enc_grad_norm": self.sae_logic.info.weight_grad_norms["W_enc"],
            "W_dec_grad_norm": self.sae_logic.info.weight_grad_norms["W_dec"],
            "tokens_processed": self.sae_logic.info.n_steps * self.config.train_batch_size
        }
        return {
            k: (float(v) if not isinstance(v, wandb.Histogram) else v)
            for k, v in log_dict.items()
        }


class SAEOverseer:
    def __init__(self, config: SAEConfig, save_at="somewhere/sae_mid", save_every=1000, restore=None, sae_postfix=None):
        self.config = config
        self.key = jax.random.PRNGKey(0)
        self.sae_trainer = SAETrainer.create(config, self.key)

        if config.wandb_name:
            self.run = wandb.init(
                entity=config.wandb_name[0], project=config.wandb_name[1], config=config,
                name=f"sae{sae_postfix}" if sae_postfix is not None else None,
            )
        self.bar = tqdm(total=config.n_steps)

        self.save_at = None
        if save_at:
            save_at = Path(save_at).resolve()
            if not restore:
                logger.warning("Erasing model checkpoint at", save_at)
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
                self.mngr_restore = ocp.CheckpointManager(
                    Path(restore).resolve(),
                    ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                    options=options,
                )
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
        save_dict = self.mngr_restore.restore(
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

    @property
    def mesh(self):
        return self.sae_trainer.mesh


def compute_whitening(data: Float[Array, "batch_size d_model"]) -> Float[Array, "d_model d_model"]:
    data = data - data.mean(axis=0, keepdims=True)
    u, s, vt = jnp.linalg.svd(data, full_matrices=True)
    return vt.astype(jnp.bfloat16).T


def main(*, restore: bool = False,
         seq_mode = "img", block_type = "double", layer = 18,
         train_mode: bool = True, save_image_activations = True,
         restore_from: str | None = None,
         stop_steps=30_000,
         **extra_config_items,):
    logger.info("Loading dataset")
    prompts_dataset = load_dataset("opendiffusionai/cc12m-cleaned")
    prompts_iterator = prompts_dataset["train"]["caption_llava_short"]
    logger.info("Creating Flux")
    ensemble = FluxEnsemble(use_schnell=True, use_fsdp=True)
    logger.info("Creating SAE trainer")
    config = SAEConfig(
        do_update=train_mode,
        seq_mode=seq_mode,
        **(dict(sae_train_every=1,
        sae_batch_size_multiplier=1)
        if not train_mode else {}),
        site = (block_type, layer),
        **extra_config_items,
    )
    if not train_mode and not restore:
        logger.warning("Enabling restore")
        restore = True
    if restore:
        stop_steps = 2_000
    sae_postfix = f"_{block_type}_l{layer}_{seq_mode}-k{config.k}"
    if config.seq_len != SAEConfig.seq_len:
        sae_postfix += f"_sl{config.seq_len}"
    if config.timesteps != SAEConfig.timesteps:
        sae_postfix += f"_t{config.timesteps}"
    save_dir = f"somewhere/sae{sae_postfix}"
    if restore_from is None:
        restore_from = save_dir
    sae_trainer = SAEOverseer(
        config,
        save_every=None if not train_mode else 1000,
        restore=restore_from if restore else None,
        save_at=save_dir,
        sae_postfix=sae_postfix
    )
    if not train_mode:
        saver = SAEOutputSaver(config, Path(f"somewhere/maxacts{sae_postfix}"), save_image_activations=save_image_activations)
    width, height = config.width_and_height
    appeared_prompts = set()
    cycle_detected = False
    activation_cache = []
    # normalization_hack = None
    
    @jax.jit
    def process_reaped(reaped):
        if block_type == "double":
            training_data = jnp.concatenate((reaped[f"double.resid.txt"], reaped[f"double.resid.img"]), axis=-2)
        else:
            training_data = reaped[f"single.resid"]
        training_data = training_data.reshape(-1, *training_data.shape[2:])  # (timesteps, sequence_length, d_model)
        training_data = config.cut_up(training_data)
        return training_data
    
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
        images, outputs = ensemble.sample(
            prompts,
            decode_latents=False,
            sample_steps=config.timesteps,
            key=key,
            width=width,
            height=height,
            return_type="debug",
            reap_double=[layer] if block_type == "double" else [],
            reap_single=[layer] if block_type == "single" else [],
        )
        reaped = outputs[1].reaped  # (timesteps, batch, sequence_length, d_model)
        assert isinstance(images, jnp.ndarray)  # to silence mypy
        logger.add(sys.stderr, level="INFO")
        training_data = process_reaped(reaped)
        take_data = int(len(training_data) * config.use_data_fraction)
        if config.transfer_to_cpu:
            training_data = np.array(training_data)
            np.random.shuffle(training_data)
            training_data = training_data[:take_data]
        else:
            training_data = jax.random.permutation(key, training_data)[:take_data]
        activation_cache.append(training_data)
        if len(activation_cache) >= config.sae_train_every:
            assert config.sae_train_every % config.sae_batch_size_multiplier == 0
            for inner_step in range(config.sae_train_every // config.sae_batch_size_multiplier):
                if train_mode:
                    if config.transfer_to_cpu:
                        cache_data = np.concatenate(activation_cache, axis=0)
                        if inner_step == 0:
                            np.random.shuffle(cache_data)
                    else:
                        if len(activation_cache) == 1:
                            cache_data = activation_cache[0]
                        else:
                            cache_data = jnp.concatenate(activation_cache, axis=0)
                        # cache_data = jax.jit(lambda x: jnp.concatenate(x, axis=0))(activation_cache)
                    if len(cache_data) == config.train_batch_size:
                        training_data, activation_cache = cache_data, []
                    else:
                        activation_cache = [cache_data[config.train_batch_size:]]
                        training_data = cache_data[:config.train_batch_size]
                    if config.transfer_to_cpu:
                        training_data = jnp.asarray(training_data)
                else:
                    assert config.sae_batch_size_multiplier == config.sae_train_every == 1
                if int(sae_trainer.sae_trainer.sae_logic.info.n_steps) == 0 and config.use_pca and config.do_update:
                    sae_trainer.sae_trainer = eqx.tree_at(
                        lambda x: x.sae_logic.info.whitening_matrix,
                        sae_trainer.sae_trainer,
                        compute_whitening(training_data)
                    )
                sae_outputs = sae_trainer.step(
                    jax.device_put(
                        training_data,
                        jax.sharding.NamedSharding(sae_trainer.mesh, jax.sharding.PartitionSpec("dp", None))))
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
        if stop_steps and step >= stop_steps:
            break


if __name__ == "__main__":
    Fire(main)
