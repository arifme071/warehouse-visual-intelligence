"""
Report
Consolidated output from all agents. Serialisable to JSON.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Report:
    detections: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    layout_suggestions: list = field(default_factory=list)
    cost_impact: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total_detections": sum(len(d) for d in self.detections),
                "total_anomalies": len(self.anomalies),
                "total_layout_suggestions": len(self.layout_suggestions),
                "estimated_daily_cost_impact_usd": self.cost_impact,
            },
            "anomalies": [asdict(a) for a in self.anomalies],
            "layout_suggestions": [asdict(s) for s in self.layout_suggestions],
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        d = self.to_dict()
        print("\n" + "=" * 50)
        print("  WAREHOUSE INTELLIGENCE REPORT")
        print("=" * 50)
        s = d["summary"]
        print(f"  Objects detected  : {s['total_detections']}")
        print(f"  Anomalies found   : {s['total_anomalies']}")
        print(f"  Layout suggestions: {s['total_layout_suggestions']}")
        print(f"  Est. daily impact : ${s['estimated_daily_cost_impact_usd']:.2f}")
        print("=" * 50 + "\n")
