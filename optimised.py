import numpy as np
import umap
from utils import compute_neighbourhood, plot_projection

def run_optimised(embeddings, index):
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=42
    )
    projected = reducer.fit_transform(embeddings)

    neighbours, _ = compute_neighbourhood(
        embeddings,
        index=index,
        k=10,
        metric="cosine"
    )

    labels = np.zeros(len(projected))
    labels[neighbours] = 1

    plot_projection(
        projected,
        labels,
        "Optimised: UMAP + Cosine"
    )

