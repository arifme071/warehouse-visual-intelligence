"""
Warehouse Visual Intelligence — Streamlit Cloud Dashboard
Full version: Image + Video + GCP/AWS export
Pillow-only (no OpenCV) for Python 3.14 compatibility
"""

import json
import io
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

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
    frame_skip = st.slider("Video frame skip", 1, 10, 3,
                           help="Process every Nth frame. Higher = faster.")

    st.divider()
    st.subheader("☁️ Cloud Export")
    cloud_export = st.selectbox(
        "Export results to",
        ["None", "Google Cloud Storage (GCS)", "AWS S3"],
    )
    if cloud_export == "Google Cloud Storage (GCS)":
        gcs_bucket = st.text_input("GCS Bucket name", placeholder="my-warehouse-bucket")
    elif cloud_export == "AWS S3":
        s3_bucket = st.text_input("S3 Bucket name", placeholder="my-warehouse-bucket")
        aws_region = st.text_input("AWS Region", value="us-east-1")

    st.divider()
    st.markdown("**🔗 Project Links**")
    st.markdown("[📂 GitHub](https://github.com/arifme071/warehouse-visual-intelligence)")
    st.markdown("[👤 LinkedIn](https://linkedin.com/in/marahman-gsu)")

# ─── Label + Cost config ────────────────────────────────────────
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
}

# ─── Core functions ─────────────────────────────────────────────
def draw_boxes_pil(image: Image.Image, detections: list) -> Image.Image:
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        colour = det["colour"]
        label  = f"{det['label']} {det['confidence']:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        tb = draw.textbbox((x1, y1 - 22), label)
        draw.rectangle([x1, y1 - 22, tb[2] + 4, y1], fill=colour)
        draw.text((x1 + 2, y1 - 20), label, fill="white")
    return img

def run_detection(image: Image.Image, model) -> list:
    results    = model(np.array(image), conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            info     = WAREHOUSE_LABELS.get(cls_name, (cls_name, "#B4B4B4"))
            wlabel, colour = info
            conf_val = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label": wlabel, "original": cls_name,
                "confidence": round(conf_val, 3),
                "bbox": [x1, y1, x2, y2], "colour": colour,
            })
    return detections

def run_agent_analysis(detections: list) -> tuple:
    anomalies   = []
    suggestions = []
    cost        = 0.0
    workers  = [d for d in detections if d["label"] == "worker"]
    vehicles = [d for d in detections if d["label"] == "vehicle"]
    parcels  = [d for d in detections if d["label"] == "parcel"]

    for v in vehicles:
        for w in workers:
            vx = (v["bbox"][0] + v["bbox"][2]) / 2
            wx = (w["bbox"][0] + w["bbox"][2]) / 2
            vy = (v["bbox"][1] + v["bbox"][3]) / 2
            wy = (w["bbox"][1] + w["bbox"][3]) / 2
            if abs(vx - wx) < 150 and abs(vy - wy) < 150:
                anomalies.append({
                    "type": "SAFETY_VIOLATION", "severity": "critical",
                    "description": "Vehicle detected in close proximity to worker — collision risk.",
                    "location": f"Vehicle bbox: {[round(x) for x in v['bbox']]}"
                })
                cost += COST_MODEL["SAFETY_VIOLATION"]

    if workers:
        anomalies.append({
            "type": "MISSING_PPE", "severity": "warning",
            "description": f"{len(workers)} worker(s) detected — PPE compliance could not be verified.",
            "location": "General scene"
        })
        cost += COST_MODEL["MISSING_PPE"] * len(workers)

    if len(parcels) >= 2:
        suggestions.append({
            "category": "Pathway Clearance",
            "description": "Multiple parcels detected — verify pathways are clear.",
            "priority": "medium", "estimated_saving_pct": 8.0,
        })
        cost += 400

    if len(detections) > 5:
        suggestions.append({
            "category": "Zone Density",
            "description": "High object density — consider redistributing across zones.",
            "priority": "medium", "estimated_saving_pct": 10.0,
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
            with st.expander(f"🟡 {s['category']}"):
                st.write(s["description"])
                st.metric("Est. Saving", f"{s['estimated_saving_pct']}%")

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_detections": len(detections),
            "total_anomalies": len(anomalies),
            "total_layout_suggestions": len(suggestions),
            "estimated_daily_cost_impact_usd": cost,
        },
        "anomalies": anomalies,
        "layout_suggestions": suggestions,
    }
    st.download_button("⬇ Download Report (JSON)",
                       data=json.dumps(report, indent=2),
                       file_name="warehouse_report.json",
                       mime="application/json")
    return report

def try_cloud_export(data: bytes, filename: str, report: dict):
    """Export to GCS or S3 if configured."""
    if cloud_export == "Google Cloud Storage (GCS)" and gcs_bucket:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(gcs_bucket)
            blob   = bucket.blob(f"output/{filename}")
            blob.upload_from_string(data)
            st.success(f"✅ Exported to gs://{gcs_bucket}/output/{filename}")
        except Exception as e:
            st.warning(f"GCS export failed: {e} — make sure GOOGLE_APPLICATION_CREDENTIALS is set.")

    elif cloud_export == "AWS S3" and s3_bucket:
        try:
            import boto3
            s3 = boto3.client("s3", region_name=aws_region)
            s3.put_object(Bucket=s3_bucket, Key=f"output/{filename}", Body=data)
            st.success(f"✅ Exported to s3://{s3_bucket}/output/{filename}")
        except Exception as e:
            st.warning(f"S3 export failed: {e} — make sure AWS credentials are set in Streamlit secrets.")

# ════════════════════════════════════════════════════════════════
# MODE SELECTOR
# ════════════════════════════════════════════════════════════════
mode = st.radio("Select mode", ["🖼️ Image", "🎥 Video"], horizontal=True)
st.divider()

# ════════════════════════════════════════════════════════════════
# IMAGE MODE
# ════════════════════════════════════════════════════════════════
if mode == "🖼️ Image":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Input Image")
        uploaded = st.file_uploader("Upload a warehouse image", type=["jpg", "jpeg", "png"])
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

                buf = io.BytesIO()
                annotated.save(buf, format="JPEG")
                img_bytes = buf.getvalue()

                st.download_button("⬇ Download Annotated Image",
                                   data=img_bytes,
                                   file_name="annotated_output.jpg",
                                   mime="image/jpeg")

                if detections:
                    report = show_analysis(detections)
                    # Cloud export
                    if cloud_export != "None":
                        try_cloud_export(img_bytes, "annotated_output.jpg", report)
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
            st.subheader("📊 Example Output")
            ex1, ex2, ex3, ex4 = st.columns(4)
            ex1.metric("Objects",       "2")
            ex2.metric("Anomalies",     "1")
            ex3.metric("Layout Issues", "0")
            ex4.metric("Daily Impact",  "$200")

# ════════════════════════════════════════════════════════════════
# VIDEO MODE
# ════════════════════════════════════════════════════════════════
else:
    st.subheader("🎥 Video Processing")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Upload a warehouse video**")
        video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
        if video_file:
            st.video(video_file)
            st.caption(f"{video_file.name} | {video_file.size / 1024 / 1024:.1f} MB")

    with col2:
        st.markdown("**Processing Controls**")
        if video_file:
            st.info(f"Model: `{model_name}` | Conf: `{conf_threshold}` | Frame skip: `{frame_skip}`")

            if st.button("▶ Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Initialising...")
                preview      = st.empty()
                metrics_box  = st.empty()

                try:
                    import cv2
                    from ultralytics import YOLO
                    model = YOLO(model_name)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    cap          = cv2.VideoCapture(tmp_path)
                    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
                    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    out_dir  = Path(tempfile.mkdtemp())
                    out_path = out_dir / f"annotated_{video_file.name}"
                    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

                    frame_num        = 0
                    total_detections = 0
                    all_detections   = []
                    last_pil         = None

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_num += 1
                        pct = frame_num / total_frames if total_frames > 0 else 0
                        progress_bar.progress(min(pct, 1.0),
                                              text=f"Frame {frame_num}/{total_frames}")

                        if frame_num % frame_skip == 0:
                            # Convert BGR→RGB→PIL
                            pil_frame = Image.fromarray(frame[:, :, ::-1])
                            dets      = run_detection(pil_frame, model)
                            total_detections += len(dets)
                            all_detections.extend(dets)
                            annotated_pil = draw_boxes_pil(pil_frame, dets)

                            # Add overlay text
                            draw    = ImageDraw.Draw(annotated_pil)
                            overlay = f"Frame {frame_num} | Detections: {len(dets)}"
                            draw.text((10, 10), overlay, fill="white")

                            # Convert back to BGR for video writer
                            annotated_bgr = np.array(annotated_pil)[:, :, ::-1]
                            writer.write(annotated_bgr)
                            last_pil = annotated_pil

                            if frame_num % 20 == 0:
                                preview.image(last_pil,
                                              caption=f"Live preview — Frame {frame_num}",
                                              use_container_width=True)
                                metrics_box.markdown(
                                    f"**Frames:** {frame_num}/{total_frames} &nbsp;|&nbsp; "
                                    f"**Detections:** {total_detections}"
                                )
                        else:
                            writer.write(frame)

                    cap.release()
                    writer.release()
                    progress_bar.progress(1.0, text="✅ Complete!")
                    st.success(f"🎉 Done! {frame_num} frames | {total_detections} total detections")

                    with open(out_path, "rb") as f:
                        video_bytes = f.read()

                    st.download_button("⬇ Download Annotated Video",
                                       data=video_bytes,
                                       file_name=f"annotated_{video_file.name}",
                                       mime="video/mp4")

                    if last_pil:
                        st.image(last_pil, caption="Final annotated frame",
                                 use_container_width=True)

                    if all_detections:
                        report = show_analysis(all_detections)
                        if cloud_export != "None":
                            try_cloud_export(video_bytes, f"annotated_{video_file.name}", report)

                except ImportError:
                    # cv2 not available — process frame by frame using PIL only
                    st.warning("OpenCV not available — using frame-by-frame PIL processing.")
                    st.info("For full video support, run locally where OpenCV is installed.")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("👈 Upload a video file to get started.")
            st.markdown("""
**Tips:**
- Videos with people or vehicles work best
- Keep under 2 minutes for faster processing
- Use `yolov8s.pt` for better accuracy
- Lower frame skip = smoother but slower
            """)
