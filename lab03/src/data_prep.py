import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# Global Rules: Set seeds for reproducibility
# ==========================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. Define Transforms
# ==========================================
# Normalize with mean=0.2860 and std=0.3530 (specific to Fashion-MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# ==========================================
# 2. Download / Load Datasets
# ==========================================
# Global rule: Data folder should be at repo root
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
os.makedirs(data_dir, exist_ok=True)

print("Downloading/Loading Fashion-MNIST...")
# Load the full 60k training dataset
full_train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

# Load the official 10k test dataset
test_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

# ==========================================
# 3. Split Training Data (55k train, 5k val)
# ==========================================
train_size = 55000
val_size = 5000

# Use a generator with seed 42 for reproducible splitting
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)

print(f"Full train dataset size: {len(full_train_dataset)}")
print(f"Split Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# ==========================================
# 4. Create DataLoaders (Batch Size 128)
# ==========================================
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 5. Sanity Checks
# ==========================================
print("\n--- Sanity Checks ---")
# Check a single batch from train_loader
images, labels = next(iter(train_loader))
print(f"Batch images shape: {images.shape} (Expected: [128, 1, 28, 28])")
print(f"Batch labels shape: {labels.shape} (Expected: [128])")
print(f"Image value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
print("✅ Task 1 Data Preparation Complete!")