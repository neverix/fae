import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .vae import FluxVAE
import jax.numpy as jnp
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

# Load the pre-trained VAE model
device = jax.devices("cpu")[0]
with jax.default_device(device):
    vae = FluxVAE("somewhere/taef1/taef1_encoder.onnx", "somewhere/taef1/taef1_decoder.onnx")

# Iterate through batches
for batch_idx, (images, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}")
    print(f"Images shape: {images.shape}")  # Should be (4, 3, 512, 512) for RGB images
    print(f"Images max: {images.max()}, min: {images.min()}")
    print(f"Labels: {labels}")
    with jax.default_device(device):
        encoded = vae.encode(jnp.asarray(images.numpy()))
        print(f"Encoded shape: {encoded.shape}")
        print(f"Encoded mean: {encoded.mean()}, std: {encoded.std()}")
