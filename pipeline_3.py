#!/usr/bin/env python3
import numpy as np
from sklearn.decomposition import PCA
import umap
import os

# ---------------------------
# Paths
# ---------------------------
FEATURES_PATH = "outputs/features.npy"    # raw 512-dim features
LABELS_PATH   = "outputs/labels.npy"
OUT_PATH      = "outputs/pipeline3_umap_euclidean.npy"

# ---------------------------
# Load data
# ---------------------------
print("Loading raw features...")
embeddings_raw = np.load(FEATURES_PATH)   # <-- this fixes your error
labels = np.load(LABELS_PATH)
print("Raw feature shape:", embeddings_raw.shape)

# ---------------------------
# PCA → 50D
# ---------------------------
print("Running PCA → 50D...")
pca_step = PCA(n_components=50, random_state=42)
pca_embeddings_50 = pca_step.fit_transform(embeddings_raw)
print("PCA output:", pca_embeddings_50.shape)

# ---------------------------
# UMAP → 2D (Euclidean)
# ---------------------------
print("Running UMAP → 2D (Euclidean)...")
umap_reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

umap_2d = umap_reducer.fit_transform(pca_embeddings_50)
print("UMAP output:", umap_2d.shape)

# ---------------------------
# Save output
# ---------------------------
os.makedirs("outputs", exist_ok=True)
np.save(OUT_PATH, umap_2d)
print("Saved:", OUT_PATH)

