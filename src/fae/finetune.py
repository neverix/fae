from numpy import add
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .vae import FluxVAE
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
# Replace 'path/to/your/folder' with the actual path to your folder containing images
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
    alpha: float

    def __init__(self, module: VLinear, rank=16, std=0.01, alpha=1., *, key):
        self.original = module
        batch_shapes = module.weight.shape[:-2]
        self.a = jax.random.normal(key, batch_shapes + (module.in_channels, rank)) * std
        self.b = jnp.zeros(batch_shapes + (rank, module.out_channels))
        self.alpha = alpha

    def __call__(self, x):
        return self.original(x) + jnp.dot(jnp.dot(x, self.a), self.b) * self.alpha

def add_lora(l):
    if not isinstance(l, VLinear):
        return l
    return Lora(l, key=next(keygen))

flux = eqx.tree_at(lambda x: x.model, flux,
    replace_fn=lambda m: jax.tree.map(add_lora, m, is_leaf=lambda x: isinstance(x, VLinear)))

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
