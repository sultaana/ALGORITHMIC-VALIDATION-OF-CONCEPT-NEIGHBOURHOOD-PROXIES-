import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Jaccard overlap matrix from your results
jaccard_matrix = np.array([
    [1.0, 0.4, 0.007, 0.001],
    [0.4, 1.0, 0.008, 0.001],
    [0.007, 0.008, 1.0, 0.001],
    [0.001, 0.001, 0.001, 1.0]
])

pipelines = ["Pipeline 1\nPCA-Euclidean",
             "Pipeline 2\nPCA-Cosine",
             "Pipeline 3\nUMAP-Euclidean",
             "Pipeline 4\nUMAP-Cosine"]

plt.figure(figsize=(6, 5))
sns.heatmap(jaccard_matrix, xticklabels=pipelines, yticklabels=pipelines, annot=True, fmt=".3f", cmap="viridis")
plt.title("Pairwise k-NN Jaccard Overlaps")
plt.tight_layout()
plt.show()

