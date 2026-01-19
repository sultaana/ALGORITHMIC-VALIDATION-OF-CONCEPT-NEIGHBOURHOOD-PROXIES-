import numpy as np

labels = np.load("human_labels.npy", allow_pickle=True)

print("Type before item():", type(labels))
print("Array shape:", labels.shape)

labels = labels.item() 

print("Type after item():", type(labels))

# Inspect contents safely
if isinstance(labels, dict):
    first_items = list(labels.items())[:10]
    print("First 10 annotations:", first_items)
    print("Total annotated samples:", len(labels))
else:
    print(labels)

