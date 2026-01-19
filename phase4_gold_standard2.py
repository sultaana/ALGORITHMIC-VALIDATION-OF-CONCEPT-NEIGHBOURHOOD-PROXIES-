import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
# Load human annotations
human_data = np.load("human_labels.npy", allow_pickle=True).item()

image_names = list(human_data.keys())
y_true = list(human_data.values())

print(f"Loaded {len(y_true)} human annotations")
# Convert filenames to indices
annotated_indices = [
    int(name.split("_")[1].split(".")[0]) for name in image_names
]

# Load embeddings (choose pipeline)
embeddings = np.load("outputs/pipeline4_umap_cosine.npy")

# Keep only annotated samples
embeddings = embeddings[annotated_indices]

print("Filtered embeddings shape:", embeddings.shape)

# k-NN within annotated space
k = 10
knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
knn.fit(embeddings)

indices = knn.kneighbors(embeddings, return_distance=False)

# Borrowed labels
borrowed_labels = []

for i in range(len(indices)):
    neighbours = indices[i][1:]  # exclude self
    neighbour_labels = [y_true[j] for j in neighbours]

    # majority vote
    borrowed = max(set(neighbour_labels), key=neighbour_labels.count)
    borrowed_labels.append(borrowed)

# Evaluate ARI
ari = adjusted_rand_score(y_true, borrowed_labels)

print("Adjusted Rand Index (ARI):", ari)
