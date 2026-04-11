"""
Cost Agent
Translates anomalies and layout inefficiencies into estimated
monetary cost impact (USD/day or USD/month).
"""

from loguru import logger


# Cost model constants (adjustable per warehouse profile)
COST_MODEL = {
    "SAFETY_VIOLATION": {
        "daily_risk_usd": 500.0,   # potential incident cost / probability
        "description": "Safety incidents cost avg $500/day in liability + downtime",
    },
    "IDLE_EQUIPMENT": {
        "daily_cost_usd": 120.0,   # equipment idle cost per unit per day
        "description": "Idle forklift costs ~$120/day in lost productivity",
    },
    "MISSING_PPE": {
        "daily_risk_usd": 200.0,
        "description": "PPE non-compliance risk: $200/day potential fine exposure",
    },
    "layout_saving_per_pct": 50.0,  # $50/day per 1% layout efficiency gain
}


class CostAgent:
    def __init__(self):
        logger.info("CostAgent ready")

    def estimate(self, anomalies: list, layout_suggestions: list) -> float:
        """
        Estimate total daily cost impact from anomalies and layout issues.

        Args:
            anomalies: List of Anomaly objects
            layout_suggestions: List of LayoutSuggestion objects

        Returns:
            Estimated daily cost impact in USD
        """
        total = 0.0

        for anomaly in anomalies:
            cost_entry = COST_MODEL.get(anomaly.type)
            if cost_entry:
                daily = cost_entry.get("daily_risk_usd") or cost_entry.get("daily_cost_usd", 0)
                total += daily
                logger.debug(f"  Anomaly [{anomaly.type}]: +${daily:.2f}/day")

        for suggestion in layout_suggestions:
            saving = suggestion.estimated_saving_pct * COST_MODEL["layout_saving_per_pct"]
            total += saving
            logger.debug(f"  Layout [{suggestion.category}]: +${saving:.2f}/day potential saving")

        return round(total, 2)

    def format_summary(self, cost: float) -> dict:
        """Return a human-readable cost breakdown."""
        return {
            "daily_usd": cost,
            "weekly_usd": round(cost * 5, 2),
            "monthly_usd": round(cost * 22, 2),
            "annual_usd": round(cost * 260, 2),
        }
