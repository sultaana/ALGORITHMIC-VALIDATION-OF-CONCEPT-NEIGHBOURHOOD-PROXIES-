import numpy as np
import umap
import matplotlib.pyplot as plt

# Load features (adjust path if needed)
features = np.load("outputs/features.npy")

# UMAP with cosine distance
umap_cosine = umap.UMAP(
    n_components=2,
    metric="cosine",
    random_state=42
)

embedding_2d = umap_cosine.fit_transform(features)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=1, alpha=0.6)
plt.title("2D UMAP Projection (Cosine Distance)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()

