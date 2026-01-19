import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# Load human annotations
human_data = np.load("human_labels.npy", allow_pickle=True).item()
image_names = list(human_data.keys())
y_true = list(human_data.values())

annotated_indices = [
    int(name.split("_")[1].split(".")[0]) for name in image_names
]

# Load embeddings
embeddings = np.load("outputs/pipeline4_umap_cosine.npy")
embeddings = embeddings[annotated_indices]

k = 10
knn = NearestNeighbors(n_neighbors=k)
knn.fit(embeddings)
indices = knn.kneighbors(embeddings, return_distance=False)

correct = 0

for i in range(len(indices)):
    neighbours = indices[i][1:]
    neighbour_labels = [y_true[j] for j in neighbours]

    borrowed = Counter(neighbour_labels).most_common(1)[0][0]

    if borrowed == y_true[i]:
        correct += 1

accuracy = correct / len(y_true)

print("Concept Borrowing Accuracy:", accuracy)
