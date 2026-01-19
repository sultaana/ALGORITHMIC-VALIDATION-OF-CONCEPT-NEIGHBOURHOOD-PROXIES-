import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_embeddings(path):
    return np.load(path)

def compute_neighbourhood(embeddings, index, k, metric):
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors([embeddings[index]])
    return indices[0], distances[0]

def plot_projection(points, labels, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=40)
    plt.title(title)
    plt.tight_layout()
    plt.show()

