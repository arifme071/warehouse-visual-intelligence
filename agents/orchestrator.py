"""
Orchestrator Agent
Coordinates all sub-agents and assembles the final report.
"""

from loguru import logger
from agents.vision_agent import VisionAgent
from agents.layout_agent import LayoutAgent
from agents.anomaly_agent import AnomalyAgent
from agents.cost_agent import CostAgent
from agents.report import Report


class Orchestrator:
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.layout_agent = LayoutAgent()
        self.anomaly_agent = AnomalyAgent()
        self.cost_agent = CostAgent()
        logger.info("Orchestrator initialised with all agents")

    def run(self, images: list) -> Report:
        """
        Run the full pipeline across all agents.

        Args:
            images: List of preprocessed image arrays

        Returns:
            Report: Consolidated report from all agents
        """
        all_detections = []
        all_anomalies = []
        all_layout_suggestions = []
        total_cost_impact = 0.0

        for idx, image in enumerate(images):
            logger.info(f"Processing image {idx + 1}/{len(images)}")

            # Step 1: Detect objects
            detections = self.vision_agent.detect(image)
            all_detections.append(detections)
            logger.debug(f"  Vision: {len(detections)} objects detected")

            # Step 2: Check for anomalies
            anomalies = self.anomaly_agent.analyze(image, detections)
            all_anomalies.extend(anomalies)
            logger.debug(f"  Anomalies: {len(anomalies)} found")

            # Step 3: Layout suggestions
            suggestions = self.layout_agent.suggest(detections)
            all_layout_suggestions.extend(suggestions)
            logger.debug(f"  Layout: {len(suggestions)} suggestions")

            # Step 4: Cost impact
            cost = self.cost_agent.estimate(anomalies, suggestions)
            total_cost_impact += cost
            logger.debug(f"  Cost impact: ${cost:.2f}")

        report = Report(
            detections=all_detections,
            anomalies=all_anomalies,
            layout_suggestions=all_layout_suggestions,
            cost_impact=total_cost_impact,
        )

        logger.success(f"Pipeline complete. Total cost impact: ${total_cost_impact:.2f}")
        return report
