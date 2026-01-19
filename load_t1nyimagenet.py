#!/usr/bin/env python3
"""
load_tinyimagenet.py
Phases:
 - Phase 1: Verify dataset load
 - Phase 2: Extract ResNet-18 features (CPU)
 - Phase 3: PCA (IncrementalPCA)
 - Phase 4: UMAP projection
Outputs:
 - ./outputs/features.npy       (float32) shape (N, feat_dim)
 - ./outputs/labels.npy         (int)     shape (N,)
 - ./outputs/pca_features.npy   (float32) shape (N, n_components)
 - ./outputs/umap_embedding.npy (float32) shape (N, 2)
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
DATASET_PATH = "/Users/mac/Desktop/config/tiny-imagenet-200"
TRAIN_ROOT = os.path.join(DATASET_PATH, "train")
BATCH_SIZE = 64           # adjust for memory / speed
NUM_WORKERS = 4           # set 0 if you have issues
OUTPUT_DIR = "./outputs"
PCA_COMPONENTS = 256      # reduce before UMAP
UMAP_N_COMPONENTS = 2
DEVICE = "cpu"            # using CPU per your setup

# Toggle phases (run one or more)
TEST_LOAD = True
EXTRACT_FEATURES = True
RUN_PCA = True
RUN_UMAP = True

# ---------------------------
def phase1_test_load():
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform)
    print("Loaded:", len(train_dataset), "training images")
    print("Classes:", len(train_dataset.classes))
    # Print first 5 class names
    print("First 5 classes:", train_dataset.classes[:5])
    return train_dataset

def phase2_extract_features(dataset):
    import torch
    from torchvision import models, transforms
    from torch.utils.data import DataLoader

    device = torch.device(DEVICE)
    # Pretrained ResNet-18, remove final fc
    model = models.resnet18(pretrained=True)
    # remove the last classification layer; get the penultimate feature vector (512-d)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features = []
    labels = []

    with torch.no_grad():
        for images, target in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            out = model(images)            # (B, 512)
            out = out.cpu().numpy()
            features.append(out.astype(np.float32))
            labels.append(target.numpy().astype(np.int32))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    np.save(os.path.join(OUTPUT_DIR, "features.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    print("Saved features:", features.shape, "labels:", labels.shape)
    return features, labels

def phase3_pca(features, n_components=PCA_COMPONENTS):
    from sklearn.decomposition import IncrementalPCA
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
    # fit in chunks if necessary
    ipca.fit(features)
    feats_pca = ipca.transform(features)
    np.save(os.path.join(OUTPUT_DIR, f"pca_features.npy"), feats_pca.astype(np.float32))
    print("Saved PCA features:", feats_pca.shape)
    return feats_pca

def phase4_umap(features, n_components=UMAP_N_COMPONENTS):
    import umap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # If numba isn't available, UMAP will still run (but slower)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(features)
    np.save(os.path.join(OUTPUT_DIR, f"umap_embedding.npy"), embedding.astype(np.float32))
    print("Saved UMAP embedding:", embedding.shape)
    return embedding

def main():
    dataset = None
    features = None

    if TEST_LOAD:
        print("Phase 1 — dataset load test")
        dataset = phase1_test_load()

    if EXTRACT_FEATURES:
        if dataset is None:
            # ensure we have a dataset
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
            ])
            dataset = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform)

        print("Phase 2 — extracting features")
        features, labels = phase2_extract_features(dataset)

    if RUN_PCA:
        if features is None:
            features = np.load(os.path.join(OUTPUT_DIR, "features.npy"))
        print("Phase 3 — running PCA")
        feats_pca = phase3_pca(features, n_components=PCA_COMPONENTS)
    else:
        feats_pca = None

    if RUN_UMAP:
        to_embed = feats_pca if feats_pca is not None else features
        print("Phase 4 — running UMAP")
        embedding = phase4_umap(to_embed, n_components=UMAP_N_COMPONENTS)

if __name__ == "__main__":
    main()

