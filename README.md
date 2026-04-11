# 🏭 Warehouse Visual Intelligence System

> A multi-agent AI system for real-time warehouse monitoring, safety detection,
> layout optimisation, and operational cost reduction using computer vision and Google Cloud + AWS.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/arifme071/warehouse-visual-intelligence-Google-and-AWS-cloud/actions/workflows/ci.yml/badge.svg)

---

## 🎯 What It Does

Upload a warehouse image (or connect a camera feed) and the system:

1. **Detects objects** — forklifts, pallets, workers, vehicles via YOLOv8
2. **Flags anomalies** — safety violations, idle equipment, missing PPE
3. **Suggests layout improvements** — zone optimisation, pathway clearance
4. **Estimates cost impact** — translates every issue into $/day savings

---

## 🤖 Multi-Agent Architecture

```
                        ┌─────────────────────┐
                        │     Orchestrator     │
                        └──────────┬──────────┘
               ┌──────────┬────────┴────────┬──────────┐
          ┌────▼───┐ ┌────▼───┐ ┌────▼────┐ ┌────▼────┐
          │ Vision │ │ Layout │ │Anomaly  │ │  Cost   │
          │ Agent  │ │ Agent  │ │ Agent   │ │ Agent   │
          └────────┘ └────────┘ └─────────┘ └─────────┘
```

| Agent | Responsibility |
|---|---|
| **Vision Agent** | Object detection (YOLOv8 / GCP Vision API / AWS Rekognition) |
| **Layout Agent** | Spatial analysis, zone optimisation suggestions |
| **Anomaly Agent** | Safety violations, misplaced items, PPE checks |
| **Cost Agent** | $/day impact from inefficiencies and risks |
| **Orchestrator** | Coordinates all agents, assembles final report |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Cloud Vision (GCP) | Google Cloud Vision API |
| Cloud Vision (AWS) | AWS Rekognition |
| Storage | Google Cloud Storage (GCS) / AWS S3 |
| Agent Framework | CrewAI + LangChain |
| LLM Backbone | Claude (Anthropic) |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov |

---

## 🚀 Quickstart

### 1. Clone and set up environment
```bash
git clone https://github.com/arifme071/warehouse-visual-intelligence-Google-and-AWS-cloud
cd warehouse-visual-intelligence-Google-and-AWS-cloud
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your GCP / AWS / Anthropic API keys
```

### 3. Run on a local image
```bash
python main.py --input data/sample_images/warehouse_01.jpg
```

### 4. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 5. (Optional) Set up Google Cloud Storage
```bash
python -m cloud_infra.setup_gcs
```

### 6. (Optional) Set up AWS S3
```bash
python -m cloud_infra.setup_aws
```

---

## 📁 Project Structure

```
warehouse-visual-intelligence-Google-and-AWS-cloud/
├── agents/               # AI agents (Vision, Layout, Anomaly, Cost, Orchestrator)
├── vision_pipeline/      # Image ingestion and preprocessing
├── cloud_infra/          # GCS + AWS S3 setup and upload helpers
├── dashboard/            # Streamlit UI
├── data/sample_images/   # Test images
├── tests/                # pytest unit tests
├── .github/workflows/    # GitHub Actions CI
├── main.py               # CLI entry point
├── requirements.txt
└── .env.example
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=agents --cov=vision_pipeline
```

---

## 📊 Sample Output

```json
{
  "summary": {
    "total_detections": 8,
    "total_anomalies": 2,
    "total_layout_suggestions": 3,
    "estimated_daily_cost_impact_usd": 1350.00
  },
  "anomalies": [
    {
      "type": "SAFETY_VIOLATION",
      "severity": "critical",
      "description": "Forklift detected in close proximity to worker."
    }
  ]
}
```

---

## 🗺️ Roadmap

- [x] YOLOv8 local inference
- [x] Multi-agent pipeline
- [x] Google Cloud Storage integration
- [x] Streamlit dashboard
- [x] GitHub Actions CI/CD
- [ ] AWS S3 + Rekognition integration
- [ ] Real-time RTSP camera feed support
- [ ] Fine-tuned PPE detection model
- [ ] BigQuery analytics + trend dashboard
- [ ] Vertex AI deployment
- [ ] Slack / email alerting

---

## 📝 License

MIT — free to use, modify, and share.

---

## 🙋 Author

Built by [Arifur Rahman](https://linkedin.com/in/arifme071) as a real-world AI + cloud engineering portfolio project.

> Fields: Robotics · Manufacturing · Warehouse Operations · Process Optimisation · Computer Vision · Multi-Agent AI
