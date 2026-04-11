"""
Tests for Vision Pipeline and Agent modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """640x640 blank warehouse-like image."""
    return np.full((640, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def mock_detections():
    from agents.vision_agent import Detection
    return [
        Detection(label="forklift", confidence=0.91, bbox=[100, 150, 300, 350]),
        Detection(label="pallet",   confidence=0.87, bbox=[400, 200, 550, 320]),
        Detection(label="worker",   confidence=0.82, bbox=[120, 160, 220, 360]),
    ]


# ─── Vision Pipeline ───────────────────────────────────────────────

def test_preprocess_output_shape(dummy_image):
    from vision_pipeline.preprocess import preprocess_image
    result = preprocess_image(dummy_image, target_size=(640, 640))
    assert result.shape == (640, 640, 3)


def test_preprocess_non_square(dummy_image):
    from vision_pipeline.preprocess import preprocess_image
    non_square = np.zeros((480, 1280, 3), dtype=np.uint8)
    result = preprocess_image(non_square, target_size=(640, 640))
    assert result.shape == (640, 640, 3)


def test_compress_to_bytes(dummy_image):
    from vision_pipeline.preprocess import compress_to_bytes
    result = compress_to_bytes(dummy_image)
    assert isinstance(result, bytes)
    assert len(result) > 0


# ─── Vision Agent ──────────────────────────────────────────────────

def test_vision_agent_mock(dummy_image):
    from agents.vision_agent import VisionAgent
    agent = VisionAgent(model_path="yolov8n.pt", use_cloud=False)
    # Without a real model, falls back to mock detections
    detections = agent._mock_detections()
    assert len(detections) > 0
    assert all(hasattr(d, "label") for d in detections)


def test_detection_area():
    from agents.vision_agent import Detection
    d = Detection(label="pallet", confidence=0.9, bbox=[0, 0, 100, 100])
    assert d.area == 10000.0


# ─── Anomaly Agent ─────────────────────────────────────────────────

def test_anomaly_forklift_worker_proximity(dummy_image, mock_detections):
    from agents.anomaly_agent import AnomalyAgent
    agent = AnomalyAgent()
    anomalies = agent.analyze(dummy_image, mock_detections)
    types = [a.type for a in anomalies]
    assert "SAFETY_VIOLATION" in types


def test_no_anomaly_when_no_workers(dummy_image):
    from agents.vision_agent import Detection
    from agents.anomaly_agent import AnomalyAgent
    detections = [Detection(label="pallet", confidence=0.9, bbox=[0, 0, 100, 100])]
    agent = AnomalyAgent()
    anomalies = agent.analyze(dummy_image, detections)
    assert all(a.type != "SAFETY_VIOLATION" for a in anomalies)


# ─── Layout Agent ──────────────────────────────────────────────────

def test_layout_suggestions_returned(mock_detections):
    from agents.layout_agent import LayoutAgent
    agent = LayoutAgent()
    suggestions = agent.suggest(mock_detections)
    assert isinstance(suggestions, list)


# ─── Cost Agent ────────────────────────────────────────────────────

def test_cost_estimate_positive(mock_detections):
    from agents.anomaly_agent import AnomalyAgent, Anomaly
    from agents.layout_agent import LayoutAgent, LayoutSuggestion
    from agents.cost_agent import CostAgent

    anomalies = [Anomaly(type="SAFETY_VIOLATION", description="test", severity="critical", location="x")]
    suggestions = [LayoutSuggestion(category="test", description="test", priority="high", estimated_saving_pct=10.0)]

    agent = CostAgent()
    cost = agent.estimate(anomalies, suggestions)
    assert cost > 0


def test_cost_zero_for_empty():
    from agents.cost_agent import CostAgent
    agent = CostAgent()
    cost = agent.estimate([], [])
    assert cost == 0.0


# ─── Report ────────────────────────────────────────────────────────

def test_report_to_dict():
    from agents.report import Report
    r = Report(cost_impact=999.0)
    d = r.to_dict()
    assert d["summary"]["estimated_daily_cost_impact_usd"] == 999.0
    assert "generated_at" in d
