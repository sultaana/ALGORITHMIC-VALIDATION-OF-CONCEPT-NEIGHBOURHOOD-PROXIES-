#!/usr/bin/env python3
"""
compute_neighborhood_overlaps.py

This script computes the local neighborhood overlaps (Jaccard similarity) 
for all four pipelines and prints a summary table.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- CONFIG ---
OUTPUT_DIR = "./outputs"
K = 10  # number of neighbors for k-NN

# --- LOAD EMBEDDINGS ---
print("Loading pipeline embeddings...")
pipeline_files = {
    "Pipeline 1 (PCA, Euclidean)": f"{OUTPUT_DIR}/pca_features.npy",
    "Pipeline 2 (PCA, Cosine)": f"{OUTPUT_DIR}/pca_features.npy",  # if cosine is just metric
    "Pipeline 3 (UMAP, Euclidean)": f"{OUTPUT_DIR}/pipeline3_umap_euclidean.npy",
    "Pipeline 4 (UMAP, Cosine)": f"{OUTPUT_DIR}/pipeline4_umap_cosine.npy",
}

embeddings = {}
for name, path in pipeline_files.items():
    embeddings[name] = np.load(path)
    print(f"{name}: {embeddings[name].shape}")

# --- HELPER FUNCTIONS ---
def compute_neighbors(embedding, k=K, metric="euclidean"):
    """Compute k-NN indices for all points."""
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    return indices[:, 1:]  # skip self-match

def jaccard_overlap(neighbors1, neighbors2):
    """Compute average Jaccard overlap between two k-NN matrices."""
    overlaps = []
    for n1, n2 in zip(neighbors1, neighbors2):
        intersection = len(set(n1) & set(n2))
        union = len(set(n1) | set(n2))
        overlaps.append(intersection / union)
    return np.mean(overlaps)

# --- COMPUTE NEIGHBORS ---
print("\nComputing k-NN for all embeddings...")
neighbors_dict = {}
for name, emb in embeddings.items():
    metric = "cosine" if "Cosine" in name else "euclidean"
    neighbors_dict[name] = compute_neighbors(emb, k=K, metric=metric)

# --- COMPUTE PAIRWISE JACCARD OVERLAPS ---
print("\nComputing pairwise Jaccard overlaps...")
pipeline_names = list(pipeline_files.keys())
overlap_matrix = np.zeros((len(pipeline_names), len(pipeline_names)))

for i, name_i in enumerate(pipeline_names):
    for j, name_j in enumerate(pipeline_names):
        overlap_matrix[i, j] = jaccard_overlap(neighbors_dict[name_i], neighbors_dict[name_j])

# --- PRINT TABLE ---
print("\nPairwise k-NN Jaccard Overlap Matrix:")
print("Rows vs Columns: Pipeline 1, 2, 3, 4")
print(np.round(overlap_matrix, 3))

