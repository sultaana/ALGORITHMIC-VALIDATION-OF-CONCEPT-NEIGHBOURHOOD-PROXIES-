# -------------------------------
# 6. Automatic Summary Bar Plots
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import os

results_folder = "pipeline_results"  
pipeline_results = {
    'Pipeline': [
        'pipeline1_umap_euclidean.npy',
        'pipeline2_umap_cosine.npy',
        'pipeline3_umap_euclidean.npy',
        'pipeline4_umap_cosine.npy'
    ],
    'ARI': [0.003191, -0.001750, -0.001447, 0.002547],
    'Concept_Borrowing': [0.005034, 0.006102, 0.007018, 0.005797],
    'Runtime_sec': [0.45, 0.14, 0.14, 0.13],
    'PeakMemory_MB': [30.32, 4.51, 3.49, 0.98]
}
df = pd.DataFrame(pipeline_results)
plt.figure(figsize=(16, 12))

# ARI plot
plt.subplot(2, 2, 1)
plt.bar(df['Pipeline'], df['ARI'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Adjusted Rand Index (ARI)")
plt.ylabel("ARI Score")

# Concept Borrowing plot
plt.subplot(2, 2, 2)
plt.bar(df['Pipeline'], df['Concept_Borrowing'], color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.title("Concept-Borrowing Accuracy")
plt.ylabel("Accuracy")

# Runtime plot
plt.subplot(2, 2, 3)
plt.bar(df['Pipeline'], df['Runtime_sec'], color='salmon')
plt.xticks(rotation=45, ha='right')
plt.title("Pipeline Runtime (seconds)")
plt.ylabel("Time (s)")

# Peak Memory plot
plt.subplot(2, 2, 4)
plt.bar(df['Pipeline'], df['PeakMemory_MB'], color='orchid')
plt.xticks(rotation=45, ha='right')
plt.title("Peak Memory Usage (MB)")
plt.ylabel("Memory (MB)")

plt.tight_layout()
plt.show()

