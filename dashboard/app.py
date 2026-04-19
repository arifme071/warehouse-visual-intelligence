"""
Warehouse Visual Intelligence — Streamlit Cloud Dashboard
Uses Pillow instead of OpenCV for Python 3.14 compatibility.
"""

import json
import io
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="Warehouse Visual Intelligence",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Warehouse Visual Intelligence System")
st.caption("Multi-agent AI · YOLOv8 · Google Cloud + AWS · Real-time cost & safety analysis")

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "YOLOv8 Model",
        ["yolov8n.pt (fastest)", "yolov8s.pt (balanced)", "yolov8m.pt (accurate)"],
        index=0,
    )
    model_name = model_choice.split(" ")[0]
    conf_threshold = st.slider("Confidence threshold", 0.10, 1.0, 0.35, 0.05)
    st.divider()
    st.markdown("**🔗 Project Links**")
    st.markdown("[📂 GitHub Repo](https://github.com/arifme071/warehouse-visual-intelligence)")
    st.markdown("[👤 LinkedIn](https://linkedin.com/in/marahman-gsu)")
    st.divider()
    st.info("Upload a warehouse image to run the full AI pipeline.")

# ─── Label mapping ──────────────────────────────────────────────
WAREHOUSE_LABELS = {
    "person":     ("worker",   "#00C800"),
    "truck":      ("vehicle",  "#FF6400"),
    "car":        ("vehicle",  "#FF6400"),
    "motorcycle": ("vehicle",  "#FF6400"),
    "bicycle":    ("vehicle",  "#FF6400"),
    "suitcase":   ("parcel",   "#FFC800"),
    "backpack":   ("parcel",   "#FFC800"),
    "chair":      ("obstacle", "#DC0000"),
    "bottle":     ("item",     "#C8C8C8"),
    "handbag":    ("parcel",   "#FFC800"),
    "tie":        ("item",     "#C8C8C8"),
}

COST_MODEL = {
    "SAFETY_VIOLATION": 500.0,
    "MISSING_PPE":      200.0,
    "IDLE_EQUIPMENT":   120.0,
}

def draw_boxes_pil(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes using Pillow."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        colour = det["colour"]
        label  = f"{det['label']} {det['confidence']:.0%}"

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

        # Label background
        text_bbox = draw.textbbox((x1, y1 - 22), label)
        draw.rectangle([x1, y1 - 22, text_bbox[2] + 4, y1], fill=colour)

        # Label text
        draw.text((x1 + 2, y1 - 20), label, fill="white")

    return img

def run_detection(image: Image.Image, model) -> list:
    """Run YOLOv8 detection on a PIL image."""
    img_array = np.array(image)
    results   = model(img_array, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            info     = WAREHOUSE_LABELS.get(cls_name, (cls_name, "#B4B4B4"))
            wlabel, colour = info
            conf_val = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label":      wlabel,
                "original":   cls_name,
                "confidence": round(conf_val, 3),
                "bbox":       [x1, y1, x2, y2],
                "colour":     colour,
            })
    return detections

def run_agent_analysis(detections: list) -> tuple:
    """Lightweight built-in agent analysis."""
    anomalies   = []
    suggestions = []
    cost        = 0.0

    workers  = [d for d in detections if d["label"] == "worker"]
    vehicles = [d for d in detections if d["label"] == "vehicle"]
    parcels  = [d for d in detections if d["label"] == "parcel"]

    # Safety violation
    for v in vehicles:
        for w in workers:
            vx = (v["bbox"][0] + v["bbox"][2]) / 2
            wx = (w["bbox"][0] + w["bbox"][2]) / 2
            vy = (v["bbox"][1] + v["bbox"][3]) / 2
            wy = (w["bbox"][1] + w["bbox"][3]) / 2
            if abs(vx - wx) < 150 and abs(vy - wy) < 150:
                anomalies.append({
                    "type":        "SAFETY_VIOLATION",
                    "severity":    "critical",
                    "description": "Vehicle detected in close proximity to worker — collision risk.",
                    "location":    f"Vehicle bbox: {[round(x) for x in v['bbox']]}"
                })
                cost += COST_MODEL["SAFETY_VIOLATION"]

    # Missing PPE
    if workers:
        anomalies.append({
            "type":        "MISSING_PPE",
            "severity":    "warning",
            "description": f"{len(workers)} worker(s) detected — PPE compliance could not be verified.",
            "location":    "General scene"
        })
        cost += COST_MODEL["MISSING_PPE"] * len(workers)

    # Layout suggestions
    if len(parcels) >= 2:
        suggestions.append({
            "category":             "Pathway Clearance",
            "description":          "Multiple parcels detected — verify pathways are clear.",
            "priority":             "medium",
            "estimated_saving_pct": 8.0,
        })
        cost += 400

    if len(detections) > 5:
        suggestions.append({
            "category":             "Zone Density",
            "description":          "High object density — consider redistributing across zones.",
            "priority":             "medium",
            "estimated_saving_pct": 10.0,
        })
        cost += 500

    return anomalies, suggestions, round(cost, 2)

def show_analysis(detections: list):
    anomalies, suggestions, cost = run_agent_analysis(detections)

    st.divider()
    st.subheader("📊 Analysis")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects Detected", len(detections))
    m2.metric("Anomalies",        len(anomalies))
    m3.metric("Layout Issues",    len(suggestions))
    m4.metric("Daily Impact",     f"${cost:.0f}")

    import pandas as pd
    st.subheader("🔍 Detected Objects")
    st.dataframe(pd.DataFrame([
        {"Label": d["label"], "Class": d["original"], "Confidence": f"{d['confidence']:.1%}"}
        for d in detections
    ]), use_container_width=True, hide_index=True)

    if anomalies:
        st.subheader("⚠️ Anomalies")
        for a in anomalies:
            icon = "🔴" if a["severity"] == "critical" else "🟡"
            with st.expander(f"{icon} {a['type']} — {a['severity'].upper()}"):
                st.write(a["description"])
                st.caption(a["location"])
    else:
        st.success("✅ No anomalies detected")

    if suggestions:
        st.subheader("📐 Layout Suggestions")
        for s in suggestions:
            icon = "🔴" if s["priority"] == "high" else "🟡"
            with st.expander(f"{icon} {s['category']}"):
                st.write(s["description"])
                st.metric("Est. Saving", f"{s['estimated_saving_pct']}%")

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_detections":              len(detections),
            "total_anomalies":               len(anomalies),
            "total_layout_suggestions":      len(suggestions),
            "estimated_daily_cost_impact_usd": cost,
        },
        "anomalies":          anomalies,
        "layout_suggestions": suggestions,
    }
    st.download_button(
        "⬇ Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="warehouse_report.json",
        mime="application/json",
    )

# ════════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Image")
    uploaded = st.file_uploader(
        "Upload a warehouse image",
        type=["jpg", "jpeg", "png"],
    )
    image_pil = None

    if uploaded:
        image_pil = Image.open(uploaded).convert("RGB")
        st.image(image_pil, caption=uploaded.name, use_container_width=True)

with col2:
    st.subheader("🤖 Detection Results")

    if image_pil is not None:
        if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):

            with st.spinner("Loading YOLOv8 model..."):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_name)
                except Exception as e:
                    st.error(f"Model load failed: {e}")
                    st.stop()

            with st.spinner("Running detection..."):
                detections = run_detection(image_pil, model)
                annotated  = draw_boxes_pil(image_pil, detections)

            st.image(annotated, caption="Annotated Output", use_container_width=True)

            # Download annotated image
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG")
            st.download_button(
                "⬇ Download Annotated Image",
                data=buf.getvalue(),
                file_name="annotated_output.jpg",
                mime="image/jpeg",
            )

            if detections:
                show_analysis(detections)
            else:
                st.info("No objects detected. Try lowering the confidence threshold or switch to yolov8s.pt.")
    else:
        st.info("👈 Upload a warehouse image to get started.")
        st.markdown("""
**Tips for best results:**
- Images with people, forklifts or vehicles work best
- Try confidence at `0.20` for challenging images
- Switch to `yolov8s.pt` for better accuracy
        """)

        # Show example output
        st.subheader("📊 Example Output")
        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Objects",      "2")
        ex2.metric("Anomalies",    "1")
        ex3.metric("Layout Issues","0")
        ex4.metric("Daily Impact", "$200")
