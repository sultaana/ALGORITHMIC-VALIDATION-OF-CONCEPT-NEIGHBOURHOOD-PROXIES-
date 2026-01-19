# full_umap_embedding_pipeline.py
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
import time
import os

# --------------------------
# 1. Load your data
# --------------------------
# Replace this with your actual feature matrix
# Example: features.shape = (num_images, num_features)
features = np.load("features.npy")  # your feature vectors for 9000 images
print("Loaded features:", features.shape)

# Optional: standardise features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --------------------------
# 2. Define pipelines
# --------------------------
pipelines = {
    "pipeline1_umap_euclidean.npy": {"metric": "euclidean", "n_neighbors": 15, "min_dist": 0.1},
    "pipeline2_umap_cosine.npy": {"metric": "cosine", "n_neighbors": 15, "min_dist": 0.1},
    "pipeline3_umap_euclidean.npy": {"metric": "euclidean", "n_neighbors": 30, "min_dist": 0.0},
    "pipeline4_umap_cosine.npy": {"metric": "cosine", "n_neighbors": 30, "min_dist": 0.0},
}

# --------------------------
# 3. Run UMAP for each pipeline and save embeddings
# --------------------------
for pipeline_file, params in pipelines.items():
    print(f"\nRunning {pipeline_file} ...")
    start_time = time.time()
    
    reducer = umap.UMAP(
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params["metric"],
        random_state=42
    )
    
    embeddings = reducer.fit_transform(features_scaled)
    
    # Save embeddings
    np.save(pipeline_file, embeddings)
    
    end_time = time.time()
    runtime = end_time - start_time
    memory_used = embeddings.nbytes / (1024**2)  # MB
    
    print(f"Done! Embeddings shape: {embeddings.shape}")
    print(f"Runtime: {runtime:.2f} seconds, Memory used: {memory_used:.2f} MB")

print("\n✅ All pipelines processed and embeddings saved!")
