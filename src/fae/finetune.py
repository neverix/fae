import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .vae import FluxVAE
import jax.numpy as jnp
from src.fae.ensemble import FluxEnsemble
from src.fae.diformer import SequentialScan
from .quant import QuantMatrix
from dataclasses import replace
from loguru import logger
import equinox as eqx
from functools import partial
import wandb
import lorax
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
# Replace 'path/to/your/folder' with the actual path to your folder containing images
dataset = datasets.ImageFolder(root="somewhere/finetune_data", transform=transform,)

# DataLoader to iterate through the dataset in batches of size 4
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# Load the Flux model
ensemble = FluxEnsemble(use_schnell=False, use_fsdp=True)
flux = ensemble.flux
is_arr = lambda x: isinstance(x, qax.primitives.ArrayValue)

def fix_lw(pytree):
    def fixer(arr):
        if isinstance(arr, (lorax.LoraWeight, QuantMatrix)):
            arr = replace(arr, shape=arr.compute_shape())
        return arr
    return jax.tree.map(fixer, pytree, is_leaf=is_arr)

def init_flux(flux, rng):
    flux = fix_lw(flux)
    def create_spec(arr):
        if len(arr.shape) == 1:
            return lorax.LORA_FULL
        else:
            return 16
    spec = jax.tree.map(
        create_spec, flux,
        is_leaf=is_arr)
    return lorax.init_lora(flux, spec, rng, is_leaf=is_arr)

flux_params, flux_logic = flux.weights, flux.logic

loop_params, flux_params = eqx.partition(
    flux_params,
    lambda x: isinstance(x, SequentialScan),
    is_leaf=lambda x: isinstance(x, SequentialScan) or is_arr(x))
single_params = eqx.tree_at(lambda x: x.double_blocks, loop_params, None)
double_params = eqx.tree_at(lambda x: x.single_blocks, loop_params, None)
single_params = fix_lw(jax.vmap(init_flux, in_axes=(0, None))(single_params, jax.random.key(0)))
double_params = fix_lw(jax.vmap(init_flux, in_axes=(0, None))(double_params, jax.random.key(1)))
flux_params = init_flux(flux_params, jax.random.key(2))
flux_params = eqx.combine(flux_params, single_params, double_params, is_leaf=is_arr)


# # flux_params = jax.vmap(init_flux, in_axes=(0, None))(flux_params, jax.random.key(0))
# def init_flux(path, arr, rng):
#     if len(arr.shape) == 1:
#         spec = lorax.LORA_FULL
#     else:
#         spec = 16
#     init_lora = lorax.init_lora
#     if path[0] in (jax.tree_util.GetAttrKey(name="single_blocks"), jax.tree_util.GetAttrKey(name="double_blocks")):
#         init_lora = jax.vmap(init_lora, in_axes=(0, None, None))
#     return init_lora(arr, spec, next(rng), is_leaf=is_arr), spec
# def key_iter():
#     i = 0
#     while True:
#         yield jax.random.key(i)
# flux_params = jax.tree_util.tree_map_with_path(
#     partial(init_flux, rng=key_iter()), flux_params,
#     is_leaf=is_arr)
flux = eqx.tree_at(lambda x: x.weights, flux, flux_params)
ensemble.flux = flux

run = wandb.init(project="fluxtune", entity="neverix")

@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(flux, inputs):
    result = flux(*inputs)
    return result.loss, result

# Iterate through batches
texts = ["Background"]
for batch_idx, (images, labels) in enumerate(data_loader):
    text_batch = [texts[i] for i in labels]
    inputs = ensemble.prepare_stuff(
        text_batch, images=jnp.asarray(images.numpy()),
        image_input_kwargs=dict(already_preprocessed=True, timesteps=None))
    (loss, image_output), grads = loss_fn(flux, inputs)
    run.log({"loss": loss})
    if batch_idx % 100 == 0:
        ensemble.flux = flux
        images = ensemble.sample(text_batch, width=WIDTH, height=HEIGHT, sample_steps=20)
        for i, image in enumerate(images):
            os.makedirs("somewhere/finetune_output", exist_ok=True)
            image.save(f"somewhere/finetune_output/{batch_idx}_{i}.png")
