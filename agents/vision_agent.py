"""
Vision Agent
Detects objects in warehouse images using YOLOv8.
Supports local inference and Google Cloud Vision API fallback.
"""

import os
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2]
    area: float = 0.0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.area = (x2 - x1) * (y2 - y1)


# Warehouse-relevant YOLO class labels
WAREHOUSE_CLASSES = {
    "person": "worker",
    "forklift": "forklift",
    "truck": "vehicle",
    "box": "pallet",
    "suitcase": "parcel",
    "chair": "obstacle",
    "bottle": "small_item",
}


class VisionAgent:
    def __init__(self, model_path: str = "yolov8n.pt", use_cloud: bool = False):
        self.use_cloud = use_cloud
        self.model = None

        if not use_cloud:
            self._load_local_model(model_path)
        else:
            self._init_cloud_vision()

    def _load_local_model(self, model_path: str):
        """Load YOLOv8 model for local inference."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded: {model_path}")
        except ImportError:
            logger.warning("ultralytics not installed. Run: pip install ultralytics")
            self.model = None

    def _init_cloud_vision(self):
        """Initialise Google Cloud Vision client."""
        try:
            from google.cloud import vision
            self.cloud_client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision client initialised")
        except ImportError:
            logger.warning("google-cloud-vision not installed.")
            self.cloud_client = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run object detection on a single image.

        Args:
            image: np.ndarray (H, W, C) in BGR format

        Returns:
            List of Detection objects
        """
        if self.use_cloud:
            return self._detect_cloud(image)
        return self._detect_local(image)

    def _detect_local(self, image: np.ndarray) -> list[Detection]:
        """Run YOLOv8 locally."""
        if self.model is None:
            logger.warning("No model loaded — returning mock detections")
            return self._mock_detections()

        results = self.model(image, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                label_raw = result.names[int(box.cls)]
                label = WAREHOUSE_CLASSES.get(label_raw, label_raw)
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                if conf >= 0.4:
                    detections.append(Detection(label=label, confidence=conf, bbox=bbox))

        logger.debug(f"Detected {len(detections)} objects locally")
        return detections

    def _detect_cloud(self, image: np.ndarray) -> list[Detection]:
        """Use Google Cloud Vision API for object detection."""
        import cv2
        _, buffer = cv2.imencode(".jpg", image)
        content = buffer.tobytes()

        from google.cloud import vision
        gcs_image = vision.Image(content=content)
        response = self.cloud_client.object_localization(image=gcs_image)

        detections = []
        h, w = image.shape[:2]
        for obj in response.localized_object_annotations:
            verts = obj.bounding_poly.normalized_vertices
            x1 = verts[0].x * w
            y1 = verts[0].y * h
            x2 = verts[2].x * w
            y2 = verts[2].y * h
            detections.append(
                Detection(
                    label=obj.name.lower(),
                    confidence=obj.score,
                    bbox=[x1, y1, x2, y2],
                )
            )

        logger.debug(f"Detected {len(detections)} objects via Cloud Vision")
        return detections

    def _mock_detections(self) -> list[Detection]:
        """Return mock detections for testing without a model."""
        return [
            Detection(label="forklift", confidence=0.91, bbox=[100, 150, 300, 350]),
            Detection(label="pallet", confidence=0.87, bbox=[400, 200, 550, 320]),
            Detection(label="worker", confidence=0.82, bbox=[600, 100, 700, 400]),
        ]
