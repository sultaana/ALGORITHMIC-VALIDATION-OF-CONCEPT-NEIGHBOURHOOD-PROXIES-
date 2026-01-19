import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# Configuration
K_NEIGHBOURS = 10
RANDOM_STATE = 42

pipelines = {
    "PCA + Euclidean": np.load("outputs/pipeline1_umap_euclidean.npy"),
    "PCA + Cosine": np.load("outputs/pipeline2_umap_cosine.npy"),
    "UMAP + Euclidean": np.load("outputs/pipeline3_umap_euclidean.npy"),
    "UMAP + Cosine": np.load("outputs/pipeline4_umap_cosine.npy"),
}

# Helper functions
def compute_knn_indices(embeddings, k, metric):
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(embeddings)
    return knn.kneighbors(return_distance=False)

def compute_neighbourhood_overlap(knn_a, knn_b):
    overlaps = []
    for i in range(len(knn_a)):
        set_a = set(knn_a[i])
        set_b = set(knn_b[i])
        overlaps.append(len(set_a.intersection(set_b)) / len(set_a))
    return np.mean(overlaps)

# Compute k-NN for each pipeline

knn_results = {}

for name, embedding in pipelines.items():
    metric = "cosine" if "Cosine" in name else "euclidean"
    knn_results[name] = compute_knn_indices(
        embedding,
        k=K_NEIGHBOURS,
        metric=metric
    )

# Compute pairwise overlap matrix
pipeline_names = list(pipelines.keys())
num_pipelines = len(pipeline_names)
overlap_matrix = np.zeros((num_pipelines, num_pipelines))

for i, p1 in enumerate(pipeline_names):
    for j, p2 in enumerate(pipeline_names):
        overlap_matrix[i, j] = compute_neighbourhood_overlap(
            knn_results[p1],
            knn_results[p2]
        )

# Plot heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(
    overlap_matrix,
    xticklabels=pipeline_names,
    yticklabels=pipeline_names,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    square=True,
    cbar_kws={"label": "Average Neighbourhood Overlap"}
)
plt.title("Heatmap of Pairwise Neighbourhood Overlap Across Pipelines")
plt.tight_layout()
plt.show()
