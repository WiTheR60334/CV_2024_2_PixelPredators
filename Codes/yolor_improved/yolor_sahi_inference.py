from yolor_sahi import YolorDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import torch
import json
import os
from pathlib import Path

# Initialize model
detection_model = YolorDetectionModel(
    model_path="best_ap50.pt",
    confidence_threshold=0.5,
    image_size=640,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

# Dataset paths (update for custom dataset)
image_dir = "test_images/yolor_test.jpg"  # Path to images
output_json = "predictions.json"  # Output predictions file

# Collect predictions
predictions = []
image_id = 1  # Simple image ID counter (adjust if using COCO IDs)

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {img_path}")
        continue

    # Run sliced inference
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=320,
        slice_width=320,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Convert predictions to COCO format
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_coco_bbox()  # [x_min, y_min, width, height]
        predictions.append({
            "image_id": image_id,
            "category_id": pred.category_id + 1,  # Adjust if YOLOR uses 0-based IDs
            "bbox": bbox,
            "score": float(pred.score)
        })

    image_id += 1

# Save predictions
with open(output_json, "w") as f:
    json.dump(predictions, f)

print(f"Predictions saved to {output_json}")