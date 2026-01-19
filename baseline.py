import numpy as np
from sklearn.decomposition import PCA
from utils import compute_neighbourhood, plot_projection

def run_baseline(embeddings, index):
    reducer = PCA(n_components=2, random_state=42)
    projected = reducer.fit_transform(embeddings)

    neighbours, _ = compute_neighbourhood(
        embeddings,
        index=index,
        k=10,
        metric="euclidean"
    )

    labels = np.zeros(len(projected))
    labels[neighbours] = 1

    plot_projection(
        projected,
        labels,
        "Baseline: PCA + Euclidean"
    )

