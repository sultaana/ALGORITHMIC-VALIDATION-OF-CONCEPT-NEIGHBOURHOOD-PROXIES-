# full_pipeline_analysis.py

import numpy as np
import time
import tracemalloc
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Configurations
# -------------------------------
pipelines = [
    "pipeline1_umap_euclidean.npy",
    "pipeline2_umap_cosine.npy",
    "pipeline3_umap_euclidean.npy",
    "pipeline4_umap_cosine.npy"
]

human_labels_file = "human_labels.npy"  # your human-annotated file

# -------------------------------
# 2. Load Human Annotations
# -------------------------------
labels_array = np.load(human_labels_file, allow_pickle=True).item()
# labels_array should be a dict: {filename: label}
filenames = list(labels_array.keys())
y_true = [labels_array[f] for f in filenames]

print(f"Loaded {len(y_true)} human annotations")

# -------------------------------
# 3. Define helper functions
# -------------------------------
def concept_borrowing_accuracy(embeddings, y_true, n_neighbors=5):
    """
    Compute the fraction of neighbours sharing the same label (concept-borrowing)
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    accuracies = []
    for i, neighbors in enumerate(indices):
        # skip self
        neighbors = neighbors[1:]
        neighbor_labels = [y_true[j] for j in neighbors]
        accuracies.append(sum(1 for l in neighbor_labels if l == y_true[i]) / n_neighbors)
    return np.mean(accuracies)

def plot_heatmap(embeddings, title="Neighbour Heatmap"):
    """
    Simple 2D heatmap of pairwise distances
    """
    from sklearn.metrics import pairwise_distances
    dist_matrix = pairwise_distances(embeddings)
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()

# -------------------------------
# 4. Run Pipelines & Measure
# -------------------------------
results = []

for pipeline_file in pipelines:
    print(f"\nRunning pipeline: {pipeline_file}")
    
    # Measure runtime and memory
    start_time = time.time()
    tracemalloc.start()
    
    embeddings = np.load(pipeline_file)
    
    # If embedding > human labels, filter to annotated subset
    if embeddings.shape[0] > len(y_true):
        embeddings = embeddings[:len(y_true), :]
    
    # ARI
    nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings)
    neighbor_indices = nbrs.kneighbors(embeddings, return_distance=False)
    
    # Flatten cluster assignments (simple 1NN cluster for ARI)
    y_pred = [neighbor_indices[i][1] for i in range(len(y_true))]  # 1st neighbour as pseudo-cluster
    ari = adjusted_rand_score(y_true, y_pred)
    
    # Concept Borrowing Accuracy
    concept_acc = concept_borrowing_accuracy(embeddings, y_true)
    
    # Memory & Runtime
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    runtime_sec = end_time - start_time
    memory_mb = peak / 1024 / 1024
    
    # Save results
    results.append({
        "Pipeline": pipeline_file,
        "ARI": ari,
        "Concept_Borrowing": concept_acc,
        "Runtime_sec": runtime_sec,
        "PeakMemory_MB": memory_mb
    })
    
    # Optional: heatmap for visual check
    plot_heatmap(embeddings, title=f"Heatmap: {pipeline_file}")

# -------------------------------
# 5. Show Results
# -------------------------------
import pandas as pd
df = pd.DataFrame(results)
print("\n==== RESULTS ====")
print(df)

# Optionally save to CSV
df.to_csv("pipeline_analysis_results.csv", index=False)
print("\nResults saved to pipeline_analysis_results.csv")

