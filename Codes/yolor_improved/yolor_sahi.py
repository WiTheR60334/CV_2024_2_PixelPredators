import logging
from typing import Any, List, Optional
import numpy as np
import torch
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)

class YolorDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["torch", "yaml"])

    def load_model(self):
        try:
            from models.models import attempt_load
            from utils.torch_utils import select_device
            device_str = str(self.device) if isinstance(self.device, torch.device) else self.device
            device = select_device(device_str)
            model = attempt_load(self.model_path, map_location=device)
            model.to(device).eval()
            if device.type != 'cpu':
                model.half()
            self.set_model(model)
        except Exception as e:
            raise TypeError(f"Failed to load YOLOR model from {self.model_path}: {str(e)}")

    def set_model(self, model: Any):
        if not hasattr(model, 'names'):
            raise Exception(f"Not a YOLOR model: {type(model)}")
        model.conf = self.confidence_threshold
        self.model = model
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        from utils.general import non_max_suppression
        from utils.augmentations import letterbox
        img = letterbox(image, new_shape=self.image_size or 640, auto=True)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.model.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if self.model.device.type != 'cpu':
            img = img.half()
        with torch.no_grad():
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres=self.confidence_threshold, iou_thres=0.45)
        self._original_predictions = pred

    @property
    def num_categories(self):
        return len(self.model.names)

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return self.model.names

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None
    ):
        original_predictions = self._original_predictions
        object_prediction_list = []
        image_predictions = original_predictions[0]
        if image_predictions is not None:
            for prediction in image_predictions.cpu().detach().numpy():
                x1, y1, x2, y2, score, category_id = prediction[:6]
                bbox = [x1, y1, x2, y2]
                category_id = int(category_id)
                category_name = self.category_mapping[str(category_id)]
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"Ignoring invalid prediction with bbox: {bbox}")
                    continue
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape
                )
                object_prediction_list.append(object_prediction)
        self._object_prediction_list = object_prediction_list