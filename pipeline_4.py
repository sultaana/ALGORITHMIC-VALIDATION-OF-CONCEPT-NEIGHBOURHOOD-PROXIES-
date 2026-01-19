#!/usr/bin/env python3
import numpy as np
import umap

RAW_FEATURES = "outputs/features.npy"
OUTPUT_PATH = "outputs/pipeline4_umap_cosine.npy"

print("Loading raw features...")
features = np.load(RAW_FEATURES).astype(np.float32)
print("Raw feature shape:", features.shape)

# ----------------------------
# PCA → 50D
# ----------------------------
print("Running PCA → 50D...")
from sklearn.decomposition import IncrementalPCA
pca_step = IncrementalPCA(n_components=50, batch_size=5000)

pca_embeddings_50 = pca_step.fit_transform(features)
print("PCA output:", pca_embeddings_50.shape)

# ----------------------------
# UMAP (Cosine)
# ----------------------------
print("Running UMAP → 2D (Cosine)...")
umap_model = umap.UMAP(
    n_components=2,
    metric="cosine",
    random_state=42,
    n_neighbors=15,
    min_dist=0.1
)

umap_output = umap_model.fit_transform(pca_embeddings_50)
print("UMAP output:", umap_output.shape)

# ----------------------------
# Save
# ----------------------------
np.save(OUTPUT_PATH, umap_output)
print("Saved:", OUTPUT_PATH)

