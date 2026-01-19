# full_pipeline_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIGURATION
# -----------------------------
# Folder where your embeddings are
embedding_folder = "outputs"
# List of pipeline files
pipelines = [
    "pipeline1_umap_euclidean.npy",
    "pipeline2_umap_cosine.npy",
    "pipeline3_umap_euclidean.npy",
    "pipeline4_umap_cosine.npy"
]
# Human annotation file
human_labels_file = "human_labels.npy"  # your saved 1311 annotations
# Number of nearest neighbors for concept borrowing
k_neighbors = 5

# -----------------------------
# LOAD HUMAN LABELS
# -----------------------------
human_labels = np.load(human_labels_file, allow_pickle=True).item()
print(f"Loaded {len(human_labels)} human annotations")

# Extract lists for easier processing
image_names = list(human_labels.keys())
labels = list(human_labels.values())

# Encode string labels to numeric for ARI
le = LabelEncoder()
y_true = le.fit_transform(labels)

# -----------------------------
# RESULTS TABLES
# -----------------------------
ari_results = []
concept_borrowing_results = []
runtime_memory_results = []

# -----------------------------
# FUNCTION: Concept Borrowing
# -----------------------------
def compute_concept_borrowing(embeddings, y_true, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    total_matches = 0
    for i in range(len(embeddings)):
        neighbor_labels = [y_true[j] for j in indices[i][1:]]  # skip self
        total_matches += sum([y_true[i] == nl for nl in neighbor_labels])
    
    return total_matches / (len(embeddings) * k)

# -----------------------------
# FUNCTION: Plot Heatmap
# -----------------------------
def plot_heatmap(embeddings, labels, pipeline_name):
    plt.figure(figsize=(8,6))
    # Convert labels to numbers for heatmap
    num_labels = LabelEncoder().fit_transform(labels)
    sns.heatmap(np.corrcoef(embeddings.T), cmap='coolwarm', square=True)
    plt.title(f"Heatmap: {pipeline_name}")
    plt.tight_layout()
    plt.savefig(f"heatmap_{pipeline_name}.png")
    plt.close()
    print(f"Saved heatmap for {pipeline_name}")

# -----------------------------
# RUN PIPELINES
# -----------------------------
for pipeline_file in pipelines:
    pipeline_path = os.path.join(embedding_folder, pipeline_file)
    
    if not os.path.exists(pipeline_path):
        print(f"Warning: {pipeline_path} not found, skipping.")
        continue
    
    print(f"\nRunning pipeline: {pipeline_file}")
    
    # Track runtime and memory
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6  # MB
    
    # Load embeddings
    embeddings = np.load(pipeline_path)
    
    # -----------------------------
    # FILTER EMBEDDINGS TO ANNOTATED IMAGES
    # -----------------------------
    # Assumes embeddings are in the same order as image_names
    filtered_embeddings = []
    filtered_labels = []
    filtered_indices = []
    
    for idx, name in enumerate(image_names):
        try:
            filtered_embeddings.append(embeddings[idx])
            filtered_labels.append(labels[idx])
            filtered_indices.append(idx)
        except IndexError:
            print(f"Index mismatch for {name}, skipping.")
    
    filtered_embeddings = np.array(filtered_embeddings)
    filtered_labels = np.array(filtered_labels)
    y_true_filtered = le.transform(filtered_labels)
    
    # -----------------------------
    # COMPUTE ARI
    # -----------------------------
    # Use kNN to cluster as predicted labels
    nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(filtered_embeddings)
    distances, indices = nbrs.kneighbors(filtered_embeddings)
    # simple "predicted" clustering: neighbor's label
    y_pred = [y_true_filtered[indices[i][1]] for i in range(len(filtered_embeddings))]
    ari = adjusted_rand_score(y_true_filtered, y_pred)
    ari_results.append((pipeline_file, ari))
    print(f"Adjusted Rand Index (ARI): {ari}")
    
    # -----------------------------
    # COMPUTE CONCEPT BORROWING
    # -----------------------------
    cb = compute_concept_borrowing(filtered_embeddings, y_true_filtered, k=k_neighbors)
    concept_borrowing_results.append((pipeline_file, cb))
    print(f"Concept Borrowing: {cb}")
    
    # -----------------------------
    # PLOT HEATMAP
    # -----------------------------
    plot_heatmap(filtered_embeddings, filtered_labels, pipeline_file.replace(".npy",""))
    
    # -----------------------------
    # RECORD RUNTIME AND MEMORY
    # -----------------------------
    end_time = time.time()
    mem_after = process.memory_info().rss / 1e6
    runtime = end_time - start_time
    memory_used = mem_after - mem_before
    runtime_memory_results.append((pipeline_file, runtime, memory_used))
    print(f"Runtime: {runtime:.2f}s, Memory used: {memory_used:.2f} MB")

# -----------------------------
# FINAL REPORT
#

