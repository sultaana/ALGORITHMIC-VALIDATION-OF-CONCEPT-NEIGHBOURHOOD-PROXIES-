#!/usr/bin/env python3

import numpy as np
import umap
import matplotlib.pyplot as plt
import os

FEATURES_PATH = "outputs/pca_features.npy"
LABELS_PATH = "outputs/labels.npy"

print("Loading PCA features...")

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"PCA file not found: {FEATURES_PATH}")

features = np.load(FEATURES_PATH)
labels = np.load(LABELS_PATH)

print("Loaded PCA features:", features.shape)
print("Loaded labels:", labels.shape)

N = 5000
features_small = features[:N]
labels_small = labels[:N]

print("Using sample:", features_small.shape)

reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean"
)

embedding = reducer.fit_transform(features_small)

print("UMAP output shape:", embedding.shape)

plt.figure(figsize=(8, 8))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=labels_small,
    s=3,
    alpha=0.6
)
plt.title("UMAP Projection (5000 samples)")
plt.savefig("outputs/umap_plot.png", dpi=300)
plt.show()

np.save("outputs/umap_embedding.npy", embedding)

print("Saved UMAP embedding to outputs/umap_embedding.npy")
print("Saved plot to outputs/umap_plot.png")

