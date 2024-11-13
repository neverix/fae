from fire import Fire
from datasets import load_dataset
from more_itertools import chunked
from .ensemble import FluxEnsemble
from dataclasses import dataclass, replace
from jaxtyping import Float, Array, UInt
import jax.numpy as jnp
import equinox as eqx
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
    
    learning_rate: float = 1e-4
    ema: float = 0.999
    grad_clip_threshold: float = 1.0


@dataclass(frozen=True)
class SAEOutput:
    x_normed: Float[Array, "batch_size d_model"]
    x: Float[Array, "batch_size d_model"]
    k_weights: Float[Array, "batch_size k"]
    k_indices: UInt[Array, "batch_size k"]
    y_normed: Float[Array, "batch_size d_model"]
    y: Float[Array, "batch_size d_model"]
    loss: Float[Array, "batch_size"]


class SAEInfo(eqx.Module):
    config: SAEConfig
    n_steps: UInt[Array, ""]
    avg_norm: Float[Array, ""]
    feature_density: Float[Array, "n_features"]
    grad_clip_percent: Float[Array, ""]
    
    def norm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        return x / self.avg_norm * math.sqrt(self.d_model)

    def denorm(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        return x * self.avg_norm / math.sqrt(self.d_model)

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
        new_grad_clip_percent = jnp.sum(list(map(jnp.sum, grad_clip_percentages))) / sum(
            map(jnp.size, grad_clip_percentages)
        )
        updated_grad_clip_percent = self.grad_clip_percent * weighting_factor + new_grad_clip_percent * new_weighting_factor
        
        return replace(self,
            n_steps=self.n_steps + 1,
            avg_norm=updated_avg_norm,
            feature_density=updated_feature_density,
            grad_clip_percent=updated_grad_clip_percent
        )


class SAE(eqx.Module):
    config: SAEConfig
    info: SAEInfo

    b_pre: Float[Array, "d_model"]
    W_enc: Float[Array, "d_model n_features"]
    b_mid: Float[Array, "n_features"]
    W_dec: Float[Array, "n_features d_model"]
    b_post: Float[Array, "n_features"]

    def __init__(self, config: SAEConfig, key: jax.random.PRNGKey):
        self.config = config
        self.b_pre = jnp.zeros((config.d_model,), dtype=config.bias_dtype)
        self.W_enc = jax.nn.initializers.orthogonal(scale=1/math.sqrt(config.d_model))(
            key, (config.d_model, config.n_features), dtype=config.param_dtype)
        self.b_mid = jnp.zeros((config.n_features,), dtype=config.bias_dtype)
        self.W_dec = self.W_enc.T
        self.b_post = jnp.zeros((config.d_model,), dtype=config.bias_dtype)
        self.info = SAEInfo(
            config=config,
            n_steps=jnp.array(0, dtype=jnp.uint32),
            avg_norm=jnp.array(1.0, dtype=jnp.float32),
            feature_density=jnp.zeros((config.n_features,), dtype=jnp.float32),
            grad_clip_percent=jnp.array(0.0, dtype=jnp.float32)
        )

    def __call__(self, x: Float[Array, "batch_size d_model"]) -> Float[Array, "batch_size d_model"]:
        info = jax.lax.stop_gradient(self.info)
        x_normed = info.norm(x)
        encodings = (x_normed - self.b_pre) @ self.W_enc
        weights, indices = jax.lax.approx_max_k(encodings, self.config.k)
        decoded = (weights[..., None] * self.W_dec[indices]).sum(-1)
        y_normed = decoded + self.b_post
        y = info.denorm(y_normed)
        return SAEOutput(
            x_normed=x_normed,
            x=x,
            k_weights=weights,
            k_indices=indices,
            y_normed=y_normed,
            y=y,
            loss=jnp.where(info.n_steps > 0, jnp.mean(jnp.square(x_normed - y_normed), axis=-1), 0)
        )
    
    def apply_updates(self, updates: "SAE", past_output: SAEOutput) -> "SAE":
        updated = eqx.apply_updates(self, updates)
        


def main():
    prompts_dataset = load_dataset("k-mktr/improved-flux-prompts")
    prompts_iterator = prompts_dataset["train"]["prompt"]
    config = SAEConfig()
    sae = SAE(config, jax.random.key(0))
    ensemble = FluxEnsemble(use_schnell=True, use_fsdp=True)
    for batch_idx, prompts in enumerate(chunked(prompts_iterator, 32)):
        key = jax.random.key(batch_idx)
        images, reaped = ensemble.sample(prompts,
                                         debug_mode=True, decode_latents=False, sample_steps=1,
                                         key=key)
        print(images.shape)
        print({k: v.shape for k, v in reaped.items()})
        for v in reaped.values():
            v.delete()


if __name__ == "__main__":
    Fire(main)