"""
Anomaly Agent
Detects safety violations, misplaced items, and unusual patterns
in warehouse imagery.
"""

from loguru import logger
from dataclasses import dataclass


@dataclass
class Anomaly:
    type: str
    description: str
    severity: str  # "critical", "warning", "info"
    location: str  # description of where in the image


class AnomalyAgent:
    def __init__(self):
        logger.info("AnomalyAgent ready")

    def analyze(self, image, detections: list) -> list[Anomaly]:
        """
        Analyse detections and image for anomalies.

        Args:
            image: np.ndarray image
            detections: List of Detection objects

        Returns:
            List of Anomaly objects
        """
        anomalies = []
        anomalies.extend(self._check_safety_violations(detections))
        anomalies.extend(self._check_idle_equipment(detections))
        anomalies.extend(self._check_missing_ppe(detections))
        return anomalies

    def _check_safety_violations(self, detections: list) -> list[Anomaly]:
        """Detect forklifts in pedestrian zones."""
        anomalies = []
        forklifts = [d for d in detections if d.label == "forklift"]
        workers = [d for d in detections if d.label == "worker"]

        if forklifts and workers:
            for f in forklifts:
                for w in workers:
                    if self._iou(f.bbox, w.bbox) > 0.01:
                        anomalies.append(
                            Anomaly(
                                type="SAFETY_VIOLATION",
                                description="Forklift detected in close proximity to worker.",
                                severity="critical",
                                location=f"Forklift bbox: {[round(x) for x in f.bbox]}",
                            )
                        )
                        logger.warning("CRITICAL: Forklift-worker proximity detected")
        return anomalies

    def _check_idle_equipment(self, detections: list) -> list[Anomaly]:
        """Flag vehicles/forklifts that appear stationary in non-parking zones."""
        anomalies = []
        vehicles = [d for d in detections if d.label in ("forklift", "vehicle")]
        # Simple heuristic: if vehicle is near image edge it may be parked incorrectly
        for v in vehicles:
            x1, _, x2, _ = v.bbox
            if x1 < 50 or x2 > 950:
                anomalies.append(
                    Anomaly(
                        type="IDLE_EQUIPMENT",
                        description=f"{v.label.capitalize()} appears to be parked near a wall/edge.",
                        severity="warning",
                        location=f"bbox: {[round(x) for x in v.bbox]}",
                    )
                )
        return anomalies

    def _check_missing_ppe(self, detections: list) -> list[Anomaly]:
        """
        Placeholder: detect workers without PPE (hard hat, vest).
        Requires a fine-tuned model for full accuracy.
        """
        anomalies = []
        workers = [d for d in detections if d.label == "worker"]
        ppe_items = [d for d in detections if d.label in ("hard_hat", "vest", "helmet")]

        if len(workers) > len(ppe_items):
            anomalies.append(
                Anomaly(
                    type="MISSING_PPE",
                    description=f"{len(workers)} workers detected but only {len(ppe_items)} PPE items visible.",
                    severity="warning",
                    location="General scene",
                )
            )
        return anomalies

    def _iou(self, box_a: list, box_b: list) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
