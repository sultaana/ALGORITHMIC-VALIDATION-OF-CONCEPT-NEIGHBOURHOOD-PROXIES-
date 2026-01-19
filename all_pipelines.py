#!/usr/bin/env python3
"""
tinyneigh_pipeline.py
Full pipeline for Tiny-ImageNet:
 - Phase 1: Load dataset
 - Phase 2: Extract features
 - Phase 3: PCA reduction
 - Phase 4: UMAP embeddings
 - Pipelines 1-4
 - k-NN neighbourhood overlap analysis
 - Visual comparison plot
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import pairwise_distances
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATASET_PATH = "/Users/mac/tinyneigh-project/data/tiny-imagenet-200"
TRAIN_ROOT = os.path.join(DATASET_PATH, "train")
OUTPUT_DIR = "./outputs"
BATCH_SIZE = 64
NUM_WORKERS = 4
PCA_COMPONENTS = 256
UMAP_COMPONENTS = 2
DEVICE = "cpu"  # CPU-only for macOS Intel
K = 10  # k-NN neighbours
NUM_POINTS = 5000  # sample subset for overlap

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- PHASE 1: Load dataset ----------------
def load_dataset():
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform)
    print("Loaded:", len(train_dataset), "training images")
    print("Classes:", len(train_dataset.classes))
    print("First 5 classes:", train_dataset.classes[:5])
    return train_dataset

# ---------------- PHASE 2: Feature extraction ----------------
def extract_features(dataset):
    device = torch.device(DEVICE)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            feat = model(imgs).cpu().numpy()
            features.append(feat)
            labels.append(lbls.numpy())
    features = np.vstack(features).astype(np.float32)
    labels = np.hstack(labels)
    
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    print("Saved features:", features.shape, "labels:", labels.shape)
    return features, labels

# ---------------- PHASE 3: PCA ----------------
def run_pca(features):
    ipca = IncrementalPCA(n_components=PCA_COMPONENTS)
    pca_features = ipca.fit_transform(features)
    np.save(os.path.join(OUTPUT_DIR, "pca_features.npy"), pca_features)
    print("Saved PCA features:", pca_features.shape)
    return pca_features

# ---------------- PHASE 4: UMAP ----------------
def run_umap(features, n_components=2, metric="euclidean"):
    import umap
    reducer = umap.UMAP(n_components=n_components, metric=metric, random_state=42)
    embedding = reducer.fit_transform(features)
    return embedding

# ---------------- k-NN & overlap ----------------
def compute_neighbours(embeddings):
    np.random.seed(42)
    subset_idx = np.random.choice(len(embeddings), size=min(NUM_POINTS, len(embeddings)), replace=False)
    subset = embeddings[subset_idx]
    distances = pairwise_distances(subset, metric="euclidean")
    neighbours = np.argsort(distances, axis=1)[:, 1:K+1]
    return subset_idx, neighbours

def jaccard_overlap(neigh_a, neigh_b):
    overlaps = []
    for a, b in zip(neigh_a, neigh_b):
        intersection = len(set(a).intersection(set(b)))
        union = len(set(a).union(set(b)))
        overlaps.append(intersection / union)
    return np.mean(overlaps)

# ---------------- VISUALIZATION ----------------
def plot_embeddings(pipelines):
    plt.figure(figsize=(12, 10))
    titles = {
        "Pipeline1": "PCA + Euclidean UMAP",
        "Pipeline2": "PCA + Cosine UMAP",
        "Pipeline3": "UMAP + Euclidean",
        "Pipeline4": "UMAP + Cosine"
    }
    for i, (name, emb) in enumerate(pipelines.items(), 1):
        plt.subplot(2, 2, i)
        plt.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.6)
        plt.title(titles[name])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_pipelines_plot.png"), dpi=300)
    print(f"Saved visualization: {os.path.join(OUTPUT_DIR, 'all_pipelines_plot.png')}")
    plt.show()

# ---------------- MAIN PIPELINE ----------------
def main():
    dataset = load_dataset()
    features, labels = extract_features(dataset)
    pca_features = run_pca(features)

    # Pipeline embeddings
    pipelines = {}
    print("Running Pipeline 1 (PCA + Euclidean UMAP)...")
    pipelines["Pipeline1"] = run_umap(pca_features, n_components=2, metric="euclidean")
    np.save(os.path.join(OUTPUT_DIR, "pipeline1_umap_euclidean.npy"), pipelines["Pipeline1"])

    print("Running Pipeline 2 (PCA + Cosine UMAP)...")
    pipelines["Pipeline2"] = run_umap(pca_features, n_components=2, metric="cosine")
    np.save(os.path.join(OUTPUT_DIR, "pipeline2_umap_cosine.npy"), pipelines["Pipeline2"])

    print("Running Pipeline 3 (UMAP + Euclidean)...")
    pipelines["Pipeline3"] = run_umap(features, n_components=2, metric="euclidean")
    np.save(os.path.join(OUTPUT_DIR, "pipeline3_umap_euclidean.npy"), pipelines["Pipeline3"])

    print("Running Pipeline 4 (UMAP + Cosine)...")
    pipelines["Pipeline4"] = run_umap(features, n_components=2, metric="cosine")
    np.save(os.path.join(OUTPUT_DIR, "pipeline4_umap_cosine.npy"), pipelines["Pipeline4"])

    # Compute k-NN neighbours
    knn_data = {}
    for name, emb in pipelines.items():
        idx, neigh = compute_neighbours(emb)
        knn_data[name] = {"idx": idx, "neigh": neigh}

    # Pairwise Jaccard overlap
    print("\nPairwise k-NN Jaccard Overlap:")
    pipeline_names = list(pipelines.keys())
    for i, a in enumerate(pipeline_names):
        for j, b in enumerate(pipeline_names):
            if j <= i: continue
            idx_a, neigh_a = knn_data[a]["idx"], knn_data[a]["neigh"]
            idx_b, neigh_b = knn_data[b]["idx"], knn_data[b]["neigh"]
            common_idx = np.intersect1d(idx_a, idx_b, assume_unique=True)
            mask_a = np.isin(idx_a, common_idx)
            mask_b = np.isin(idx_b, common_idx)
            score = jaccard_overlap(neigh_a[mask_a], neigh_b[mask_b])
            print(f"{a:<10} vs {b:<10} → {score:.4f}")

    # Summary Table
    print("\nPipeline Summary Table:")
    print(f"{'Pipeline':<10} {'Dimensionality Reduction':<25} {'Similarity Metric':<15}")
    summary = {
        "Pipeline1": ("PCA", "Euclidean"),
        "Pipeline2": ("PCA", "Cosine"),
        "Pipeline3": ("UMAP", "Euclidean"),
        "Pipeline4": ("UMAP", "Cosine")
    }
    for name, (dim_red, metric) in summary.items():
        print(f"{name:<10} {dim_red:<25} {metric:<15}")

    # Plot all pipelines
    plot_embeddings(pipelines)

if __name__ == "__main__":
    main()

