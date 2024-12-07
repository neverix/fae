import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .vae import FluxVAE
import jax.numpy as jnp
from src.fae.ensemble import FluxEnsemble
from loguru import logger
import jax

# Define transformations: center crop, resize to 512x512, and convert to tensor
transform = transforms.Compose([
    transforms.CenterCrop(512),  # Center crop to 512x512
    transforms.Resize((512, 512)),  # Resize to ensure it's 512x512
    transforms.ToTensor()  # Convert to tensor
])

# Load dataset from a folder
# Replace 'path/to/your/folder' with the actual path to your folder containing images
dataset = datasets.ImageFolder(root="somewhere/finetune_data", transform=transform,)

# DataLoader to iterate through the dataset in batches of size 4
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load the Flux model
ensemble = FluxEnsemble(use_schnell=False, use_fsdp=True)

# Iterate through batches
texts = ["Background"]
for batch_idx, (images, labels) in enumerate(data_loader):
    text_batch = [texts[i] for i in labels]
    inputs = ensemble.prepare_stuff(
        text_batch, images=jnp.asarray(images.numpy()),
        image_input_kwargs=dict(already_preprocessed=True, timesteps=None))
    image_output = ensemble.flux(*inputs)
    logger.info(f"{image_output.loss}")
