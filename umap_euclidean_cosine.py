import numpy as np
import umap
import matplotlib.pyplot as plt

X = np.load("/Users/mac/tinyneigh-project/outputs/umap_embedding.npy")

# UMAP with Euclidean distance
umap_euc = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
)
embedding_euc = umap_euc.fit_transform(X)

# UMAP with Cosine distance
umap_cos = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42
)
embedding_cos = umap_cos.fit_transform(X)

# side-by-side plot 
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(embedding_euc[:, 0], embedding_euc[:, 1], s=5, alpha=0.7)
plt.title("UMAP Projection (Euclidean)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.subplot(1, 2, 2)
plt.scatter(embedding_cos[:, 0], embedding_cos[:, 1], s=5, alpha=0.7)
plt.title("UMAP Projection (Cosine)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.tight_layout()
plt.show()
