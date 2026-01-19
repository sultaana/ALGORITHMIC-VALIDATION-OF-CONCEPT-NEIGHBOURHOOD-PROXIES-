import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
DATASET_PATH = "/Users/mac/Desktop/config/tiny-imagenet-200"
IMG_SIZE = 64   # Tiny-ImageNet uses 64x64 images
BATCH_SIZE = 64

# ---------------------------------------------
# TRANSFORMS
# ---------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

# ---------------------------------------------
# DATASET LOADING
# ---------------------------------------------
train_dir = os.path.join(DATASET_PATH, "train")
val_dir = os.path.join(DATASET_PATH, "val")

# Tiny-ImageNet validation is unusual: images are all in one folder
# ImageFolder requires class folders, so we must move or restructure val annotations
# But PyTorch provides a wrapper for this; here is a simplified version

# Load TRAIN
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

# Load VAL (PyTorch expects val/class folders, but Tiny-ImageNet puts all images together.
# If your val is already reorganised, this will work. If not, tell me.)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# ---------------------------------------------
# DATA LOADERS
# ---------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------
# TEST SAMPLE
# ---------------------------------------------
print("Total training samples:", len(train_dataset))
print("Total validation samples:", len(val_dataset))

# Inspect one batch
images, labels = next(iter(train_loader))
print("Batch image tensor shape:", images.shape)
print("Batch label tensor shape:", labels.shape)

