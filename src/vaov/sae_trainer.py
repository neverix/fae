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
import optax
import jax.numpy as jnp
import equinox as eqx
import wandb
import math
import jax


@dataclass(frozen=True)
class SAEConfig:
    d_model: int = 2048
    n_features: int = 32768
    
    param_dtype: jax.typing.DTypeLike = jnp.bfloat16
    bias_dtype: jax.typing.DTypeLike = jnp.float32
    
    k: int = 32
    aux_k: int = 512
    aux_k_after: int = 64
    aux_k_coeff: float = 1/32

    batch_size: int = 32
    n_steps: int = 1_000
    wandb_name: tuple[str, str] = ("neverix", "vaov")

    tp_size: int = jax.local_device_count()
    
    learning_rate: float = 5e-4
    beta1: float = 0.0
    beta2: float = 0.99
    eps: float = 1e-8
    ema: float = 0.995
    grad_clip_threshold: float = 1.0
    warmup_steps: int = 50


class SAEOutput(eqx.Module):
    x_normed: Float[Array, "batch_size d_model"]
    x: Float[Array, "batch_size d_model"]
    k_weights: Float[Array, "batch_size k"]
    k_indices: UInt[Array, "batch_size k"]
    y_normed: Float[Array, "batch_size d_model"]
    y: Float[Array, "batch_size d_model"]
    losses: dict[str, Float[Array, "batch_size"]]
    loss: Float[Array, "batch_size"]


class SAEInfo(eqx.Module):
    config: SAEConfig = eqx.field(static=True)
    n_steps: UInt[Array, ""]
    avg_norm: Float[Array, ""]
    feature_density: Float[Array, "n_features"]
    grad_clip_percent: Float[Array, ""]
    
    def norm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        return x / self.avg_norm * math.sqrt(self.config.d_model)

    def denorm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        return x * self.avg_norm / math.sqrt(self.config.d_model)

    def step(self, updates: "SAE", outputs: SAEOutput):
        weighting_factor, new_weighting_factor = self.n_steps / (self.n_steps + 1), 1 / (self.n_steps + 1)

        new_avg_norm = jnp.mean(jnp.linalg.norm(outputs.x, axis=-1))
        updated_avg_norm = self.avg_norm * weighting_factor + new_avg_norm * new_weighting_factor
        
        new_feature_density = jnp.zeros(self.feature_density.shape).at[outputs.k_indices.flatten()].add(1)
        updated_feature_density = self.feature_density * weighting_factor + new_feature_density * new_weighting_factor
        
        flat_updates = jax.tree.flatten(updates)[0]
        grad_clip_percentages = [
            jnp.abs(v) > self.config.grad_clip_threshold for v in flat_updates
        ]
        new_grad_clip_percent = sum(map(jnp.sum, grad_clip_percentages)) / sum(map(jnp.size, grad_clip_percentages))
        updated_grad_clip_percent = self.grad_clip_percent * weighting_factor + new_grad_clip_percent * new_weighting_factor
        
        return replace(self,
            n_steps=self.n_steps + 1,
            avg_norm=updated_avg_norm,
            feature_density=updated_feature_density,
            grad_clip_percent=updated_grad_clip_percent
        )
    
    @classmethod
    def pspec(cls, config: SAEConfig) -> "SAEInfo":
        return SAEInfo(
            config=config,
            n_steps=P(),
            avg_norm=P(),
            feature_density=P("tp"),
            grad_clip_percent=P(),
        )


SPARSE_MATMUL_BATCH = 64
def sparse_matmul(
    weights: Float[Array, "batch_size k"],
    indices: UInt[Array, "batch_size k"],
    W: Float[Array, "n_features d_model"]) -> Float[Array, "batch_size d_model"]:
    def sparse_matmul_basic(wi):
        weights, indices = wi
        return (weights[:, None] * W[indices]).sum(0)
    return jax.lax.map(sparse_matmul_basic, (weights, indices), batch_size=SPARSE_MATMUL_BATCH)

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
        W_enc = jax.nn.initializers.orthogonal(scale=1 / math.sqrt(config.d_model))(
            key, (config.d_model, config.n_features), dtype=config.param_dtype
        )
        return cls(
            config=config,
            b_pre=jnp.zeros((config.d_model,), dtype=config.bias_dtype),
            W_enc=W_enc,
            b_mid=jnp.zeros((config.n_features,), dtype=config.bias_dtype),
            W_dec=W_enc.T,
            b_post=jnp.zeros((config.d_model,), dtype=config.bias_dtype),
            info=SAEInfo(
                config=config,
                n_steps=jnp.array(0, dtype=jnp.uint32),
                avg_norm=jnp.array(1.0, dtype=jnp.float32),
                feature_density=jnp.zeros((config.n_features,), dtype=jnp.float32),
                grad_clip_percent=jnp.array(0.0, dtype=jnp.float32)
            )
        )

    def __call__(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        info = jax.lax.stop_gradient(self.info)
        x_normed = info.norm(x)
        encodings = (x_normed - self.b_pre) @ self.W_enc
        weights, indices = jax.lax.approx_max_k(encodings, self.config.k)
        decoded = sparse_matmul(weights, indices, self.W_dec)
        y_normed = decoded + self.b_post
        y = info.denorm(y_normed)
        recon_loss = jnp.mean(jnp.square(x_normed - y_normed), axis=-1)
        recon_loss = jnp.where(info.n_steps > 0, recon_loss, 0)
        
        # TODO decode with aux_k
        aux_y_normed = y_normed
        aux_k_loss = jnp.mean(jnp.square(x_normed - aux_y_normed), axis=-1)
        aux_k_loss = jnp.where(info.n_steps >= self.config.aux_k_after, aux_k_loss, 0)
        
        loss = recon_loss + self.config.aux_k_coeff * aux_k_loss
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
            loss=loss
        )
    
    def apply_updates(self, updates: "SAE", past_output: SAEOutput) -> "SAE":
        updated = eqx.apply_updates(self, updates)
        updated = eqx.tree_at(lambda x: x.b_pre, updated, replace_fn=lambda b_pre:
            jnp.where(self.info.n_steps > 0, b_pre, -past_output.x_normed.mean(axis=0)))
        updated = eqx.tree_at(lambda x: x.info, updated, replace_fn=lambda info: info.step(updates, past_output))
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

class SAETrainer(eqx.Module):
    config: SAEConfig
    sae_params: SAE
    sae_logic: SAE
    mesh: jax.sharding.Mesh
    ema: SAE
    optimizer: optax.GradientTransformation
    optimizer_state: SAE
    
    @classmethod
    def create(cls, config: SAEConfig, key: jax.random.PRNGKey):
        shape_request = (-1, config.tp_size)
        device_count = jax.device_count()
        mesh_shape = np.arange(device_count).reshape(*shape_request).shape
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
        mesh = jax.sharding.Mesh(physical_mesh, ("dp", "tp"))

        def sae_creator(key):
            return SAE.create(config, key)

        pspec = SAE.pspec(config)
        sae = jax.jit(
            sae_creator,
            out_shardings=jax.tree.map(
                lambda x: jax.sharding.NamedSharding(mesh, x), pspec
            ),
        )(key)
        sae_params, sae_logic = sae.split()
        
        optimizer = optax.adam(config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.eps)
        optimizer_state = optimizer.init(sae_params)
        
        return cls(
            config=config,
            mesh=mesh,
            sae_params=sae_params,
            sae_logic=sae_logic,
            ema=jax.tree.map(jnp.copy, sae_params),
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
        
        updates, optimizer_state = self.optimizer.update(sae_grad, self.optimizer_state, self.sae_params)
        start_updating = self.sae_logic.info.n_steps > 0
        updates = jax.tree.map(
            lambda u: jnp.where(start_updating, u, 0),
            updates)
        optimizer_state = jax.tree.map(
            lambda o: jnp.where(start_updating, o, 0),
            optimizer_state
        )
        
        sae = eqx.combine(self.sae_params, self.sae_logic)
        updated_sae = sae.apply_updates(updates, sae_outputs)
        sae_params, sae_logic = updated_sae.split()
        updated_ema = jax.tree.map(lambda ema, sae: ema * self.config.ema + sae * (1 - self.config.ema),
                                   self.ema, sae_params)
        return replace(self, sae_params=sae_params, sae_logic=sae_logic,
                       ema=updated_ema, optimizer_state=optimizer_state), sae_outputs


class SAEOverseer:
    def __init__(self, config: SAEConfig):
        self.config = config
        self.key = jax.random.PRNGKey(0)
        self.sae_trainer = SAETrainer.create(config, self.key)
        if config.wandb_name:
            self.run = wandb.init(entity=config.wandb_name[0], project=config.wandb_name[1], config=config)

    def step(self, x: Float[Array, "batch_size d_model"]) -> SAEOutput:
        self.sae_trainer, sae_outputs = self.sae_trainer.step(x)
        self.run.log(dict(loss=float(sae_outputs.loss.mean())))
        return sae_outputs


def main():
    prompts_dataset = load_dataset("k-mktr/improved-flux-prompts")
    prompts_iterator = prompts_dataset["train"]["prompt"]
    config = SAEConfig()
    sae_trainer = SAEOverseer(config)
    ensemble = FluxEnsemble(use_schnell=True, use_fsdp=True)
    for batch_idx, prompts in enumerate(chunked(prompts_iterator, 32)):
        key = jax.random.key(batch_idx)
        images, reaped = ensemble.sample(prompts,
                                         debug_mode=True, decode_latents=False, sample_steps=1,
                                         key=key)
        training_data = reaped["double_img"][-1]
        training_data = training_data.astype(jnp.bfloat16).reshape(-1, 2048)
        for v in reaped.values():
            v.delete()
        sae_outputs = sae_trainer.step(training_data)
        print(sae_outputs.loss.mean())


if __name__ == "__main__":
    Fire(main)