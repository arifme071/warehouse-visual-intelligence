"""
Layout Agent
Analyses spatial distribution of detected objects and suggests
layout optimisations to reduce travel time and improve throughput.
"""

from loguru import logger
from dataclasses import dataclass


@dataclass
class LayoutSuggestion:
    category: str
    description: str
    priority: str  # "high", "medium", "low"
    estimated_saving_pct: float  # % improvement estimate


class LayoutAgent:
    def __init__(self):
        self.rules = self._load_rules()
        logger.info("LayoutAgent ready")

    def _load_rules(self) -> list:
        """
        Rule-based layout heuristics.
        Can be replaced by an ML model or LLM reasoning later.
        """
        return [
            {
                "trigger": "forklift_near_worker",
                "check": lambda d: self._forklifts_near_workers(d),
                "suggestion": LayoutSuggestion(
                    category="Safety Zone",
                    description="Forklifts detected in worker zones. Recommend separating pedestrian lanes.",
                    priority="high",
                    estimated_saving_pct=15.0,
                ),
            },
            {
                "trigger": "pallet_blocking",
                "check": lambda d: self._pallets_in_pathways(d),
                "suggestion": LayoutSuggestion(
                    category="Pathway Clearance",
                    description="Pallets detected in likely pathway zones. Move to designated storage.",
                    priority="high",
                    estimated_saving_pct=10.0,
                ),
            },
            {
                "trigger": "cluster_density",
                "check": lambda d: self._high_density_cluster(d),
                "suggestion": LayoutSuggestion(
                    category="Zone Density",
                    description="High object density detected in one zone. Redistribute to balance load.",
                    priority="medium",
                    estimated_saving_pct=8.0,
                ),
            },
        ]

    def suggest(self, detections: list) -> list[LayoutSuggestion]:
        """
        Generate layout suggestions based on detections.

        Args:
            detections: List of Detection objects from VisionAgent

        Returns:
            List of LayoutSuggestion objects
        """
        suggestions = []
        for rule in self.rules:
            if rule["check"](detections):
                suggestions.append(rule["suggestion"])
                logger.debug(f"Layout rule triggered: {rule['trigger']}")
        return suggestions

    # --- Heuristic checks ---

    def _forklifts_near_workers(self, detections: list) -> bool:
        """Check if forklifts and workers share overlapping bounding box zones."""
        forklifts = [d for d in detections if d.label == "forklift"]
        workers = [d for d in detections if d.label == "worker"]
        for f in forklifts:
            for w in workers:
                if self._overlap_or_close(f.bbox, w.bbox, threshold=100):
                    return True
        return False

    def _pallets_in_pathways(self, detections: list) -> bool:
        """Simple check: pallets detected near centre of image (likely pathway)."""
        pallets = [d for d in detections if d.label == "pallet"]
        for p in pallets:
            x1, y1, x2, y2 = p.bbox
            cx = (x1 + x2) / 2
            if 300 < cx < 700:  # rough centre-zone threshold
                return True
        return False

    def _high_density_cluster(self, detections: list) -> bool:
        """Check if more than 5 objects are packed in a small region."""
        if len(detections) < 5:
            return False
        bboxes = [d.bbox for d in detections]
        xs = [(b[0] + b[2]) / 2 for b in bboxes]
        ys = [(b[1] + b[3]) / 2 for b in bboxes]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        return x_range < 300 and y_range < 300

    def _overlap_or_close(self, bbox_a: list, bbox_b: list, threshold: float = 50) -> bool:
        """Check if two bounding boxes are within a pixel threshold."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        horiz = abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2)
        vert = abs((ay1 + ay2) / 2 - (by1 + by2) / 2)
        return horiz < threshold and vert < threshold
