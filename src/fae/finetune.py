from numpy import add
import torch
from functools import partial
import optax
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .vae import FluxVAE
from .quant import kernel_mode
from .quant import MockQuantMatrix, QuantMatrix, is_arr
import jax.numpy as jnp
from src.fae.ensemble import FluxEnsemble
from src.fae.diflayers import VLinear
from dataclasses import replace
from loguru import logger
import equinox as eqx
from functools import partial
from jaxtyping import Float, Array
import wandb
import os
import qax
import jax

# Define transformations: center crop, resize to 512x512, and convert to tensor
WIDTH, HEIGHT = 256, 256
transform = transforms.Compose([
    transforms.Resize(min(HEIGHT, WIDTH)),
    transforms.CenterCrop((WIDTH, HEIGHT)),  # Center crop to 512x512
    transforms.ToTensor()  # Convert to tensor
])

# Load dataset from a folder
dataset = datasets.ImageFolder(root="somewhere/finetune_data", transform=transform,)

# DataLoader to iterate through the dataset in batches of size 4
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# Load the Flux model
ensemble = FluxEnsemble(use_schnell=False, use_fsdp=True)
flux = ensemble.flux

def keygenerator():
    key = jax.random.key(0)
    while True:
        key, val = jax.random.split(key)
        yield val
keygen = keygenerator()

class Lora(eqx.Module):
    original: VLinear
    a: Float[Array, "n k"]
    b: Float[Array, "k m"]
    alpha: float = eqx.field(static=True)

    def __init__(self, module: VLinear, rank=16, std=0.01, alpha=1., *, key):
        self.original = module
        batch_shapes = module.weight.shape[:-2]
        dtype = module.weight.dtype
        self.a = jax.random.normal(key, batch_shapes + (module.in_channels, rank), dtype=dtype) * std
        self.b = jnp.zeros(batch_shapes + (rank, module.out_channels), dtype=dtype)
        self.alpha = alpha

    def __call__(self, x, **kwargs):
        return self.original(x) + jnp.dot(jnp.dot(x, self.a), self.b) * self.alpha

def add_lora(l):
    if not isinstance(l, VLinear):
        return l
    return Lora(l, key=next(keygen))

flux = eqx.tree_at(lambda x: x.model, flux,
    replace_fn=lambda m: jax.tree.map(add_lora, m, is_leaf=lambda x: isinstance(x, VLinear)))

run = wandb.init(project="fluxtune", entity="neverix")

# from .quant import MockQuantMatrix
# print("---", MockQuantMatrix.mockify(flux))
# import equinox._ad

# @eqx.filter_value_and_grad(has_aux=True)
# def loss_fn(flux, inputs):
#     result = flux(*inputs)
#     return result.loss, result

@partial(jax.value_and_grad, has_aux=True)
def loss_fn(flux_good, flux_bad, inputs):
    result = eqx.combine(flux_good, flux_bad)(*inputs)
    return result.loss, result

def flux_split(flux):
    return eqx.partition(
            flux,
            lambda x: is_arr(x) and (not any(c == jnp.dtype(x).kind for c in "biu")) and (not isinstance(x, (QuantMatrix, FluxVAE))),
            is_leaf=lambda x: is_arr(x) or isinstance(x, FluxVAE))

# def flux_split(flux):
#     flux = MockQuantMatrix.mockify(flux)
#     flux_good, flux_bad = eqx.partition(
#             flux,
#             lambda x: isinstance(x, Lora),
#             is_leaf=lambda x: is_arr(x) or isinstance(x, Lora))
#     def filter_original(x, for_train=False):
#         if not isinstance(x, Lora):
#             return x
#         if for_train:
#             return eqx.tree_at(lambda x: x.original, x, None)
#         else:
#             x = eqx.tree_at(lambda x: x.a, x, None)
#             x = eqx.tree_at(lambda x: x.b, x, None)
#             return x
#     def filter_tree(x, **kwargs):
#         return jax.tree.map(partial(filter_original, **kwargs), x, is_leaf=lambda x: is_arr(x) or isinstance(x, Lora))
#     flux_bad_ = filter_tree(flux_good, for_train=False)
#     flux_good = filter_tree(flux_good, for_train=True)
#     flux_bad = eqx.combine(flux_bad, flux_bad_, is_leaf=is_arr)
#     return flux_good, flux_bad

EPOCHS = 200
scheduler = optax.schedules.warmup_cosine_decay_schedule(0.0, 1e-5, 50, EPOCHS * len(data_loader))
optimizer = optax.adam(scheduler)
opt_state = eqx.filter_jit(lambda flux: optimizer.init(flux_split(flux)[0]))(flux)

# @partial(jax.jit, donate_argnums=(0, 1))
# @jax.jit
@eqx.filter_jit(donate="all-except-first")
def train_step(inputs, flux, opt_state):
    flux_good, flux_bad = flux_split(flux)
    flux_bad = MockQuantMatrix.mockify(flux_bad)
    (loss, image_output), grads = loss_fn(flux_good, flux_bad, inputs)
    updates, new_state = optimizer.update(grads, opt_state)
    updated = eqx.combine(MockQuantMatrix.unmockify(flux_bad), optax.apply_updates(flux_good, updates), is_leaf=lambda x: is_arr(x) or isinstance(x, FluxVAE))
    return (loss, image_output), new_state, updated

# Iterate through batches
texts = ["Background"]
batch_idx = 0
for _ in range(EPOCHS):
    for images, labels in data_loader:
        text_batch = [texts[i] for i in labels]
        inputs = ensemble.prepare_stuff(
            text_batch, images=jnp.asarray(images.numpy()),
            image_input_kwargs=dict(already_preprocessed=True, timesteps=None))
        (loss, image_output), opt_state, flux = train_step(inputs, flux, opt_state)
        run.log({"loss": loss})
        if batch_idx % 100 == 0:
            ensemble.flux = flux
            images = ensemble.sample(text_batch, width=WIDTH, height=HEIGHT, sample_steps=20)
            for i, image in enumerate(images):
                os.makedirs("somewhere/finetune_output", exist_ok=True)
                image.save(f"somewhere/finetune_output/{batch_idx}_{i}.png")
        batch_idx += 1
