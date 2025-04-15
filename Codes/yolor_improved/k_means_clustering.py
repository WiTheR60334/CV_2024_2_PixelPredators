import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_visdrone_boxes(annotation_folder):
    boxes = []
    for filename in os.listdir(annotation_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(annotation_folder, filename)) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        w, h = float(parts[2]), float(parts[3])
                        if w > 0 and h > 0:
                            boxes.append([w, h])
    return np.array(boxes)

path = "VisDrone2019-DET-train/VisDrone2019-DET-train/annotations"

boxes = load_visdrone_boxes(path)
print(f"Loaded {len(boxes)} bounding boxes")

k = 12  # Number of anchors
kmeans = KMeans(n_clusters=k, random_state=42).fit(boxes)
anchors = kmeans.cluster_centers_

anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
print("\n Anchor Boxes (w, h):\n", anchors)

image_size = 960
normalized = (anchors / image_size).round(3)
print("\n Normalized Anchors (YOLO format):\n", normalized)

plt.figure(figsize=(8, 6))
plt.scatter(boxes[:, 0], boxes[:, 1], alpha=0.2, label='Bounding Boxes')
plt.scatter(anchors[:, 0], anchors[:, 1], color='red', label='Anchors')
plt.xlabel('Width'), plt.ylabel('Height')
plt.title('Anchor Box Clustering - KMeans')
plt.legend(), plt.grid(True)
plt.show()