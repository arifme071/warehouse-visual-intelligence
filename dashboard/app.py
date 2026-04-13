"""
Warehouse Visual Intelligence — Streamlit Dashboard (Phase 2 + Video)
Supports both image and video processing with live YOLOv8 detections.
Run: streamlit run dashboard/app.py
"""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Warehouse Visual Intelligence",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Warehouse Visual Intelligence System")
st.caption("Multi-agent AI · YOLOv8 · Google Cloud + AWS · Real-time cost analysis")

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
    frame_skip = st.slider("Video frame skip (higher = faster)", 1, 10, 3,
                           help="Process every Nth frame. Higher = faster but less smooth.")
    cloud_export = st.selectbox("Export to cloud", ["None", "Google Cloud Storage", "AWS S3"])
    st.divider()
    st.markdown("**Quick Start**")
    st.code("streamlit run dashboard/app.py", language="bash")

WAREHOUSE_LABELS = {
    "person":     ("worker",   (0, 200, 0)),
    "truck":      ("vehicle",  (0, 100, 255)),
    "car":        ("vehicle",  (0, 100, 255)),
    "motorcycle": ("vehicle",  (0, 100, 255)),
    "bicycle":    ("vehicle",  (0, 100, 255)),
    "suitcase":   ("parcel",   (255, 200, 0)),
    "backpack":   ("parcel",   (255, 200, 0)),
    "chair":      ("obstacle", (0, 0, 220)),
    "bottle":     ("item",     (200, 200, 200)),
    "handbag":    ("parcel",   (255, 200, 0)),
}

def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        colour = det["colour"]
        label  = f"{det['label']} {det['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def run_detection(image, model):
    results = model(image, conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            info = WAREHOUSE_LABELS.get(cls_name, (cls_name, (180, 180, 180)))
            wlabel, colour = info
            conf_val = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "label": wlabel, "original": cls_name,
                "confidence": round(conf_val, 3),
                "bbox": [x1, y1, x2, y2], "colour": colour,
            })
    return detections

# ─── Mode Tabs ───────────────────────────────────────────────────
mode = st.radio("Select mode", ["🖼️ Image", "🎥 Video"], horizontal=True)
st.divider()

# ════════════════════════════════════════════════════════════════
# IMAGE MODE
# ════════════════════════════════════════════════════════════════
if mode == "🖼️ Image":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Input")
        tab_upload, tab_sample = st.tabs(["Upload Image", "Sample Images"])
        image_array = None
        image_source = None

        with tab_upload:
            uploaded = st.file_uploader("Choose a warehouse image", type=["jpg", "jpeg", "png"])
            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_source = "upload"
                st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
                         caption=uploaded.name, use_container_width=True)

        with tab_sample:
            sample_dir = Path("data/sample_images")
            samples = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png")) if sample_dir.exists() else []
            if samples:
                selected = st.selectbox("Select sample", [p.name for p in samples])
                if image_source != "upload":
                    image_array = cv2.imread(str(sample_dir / selected))
                    image_source = "sample"
                    if image_array is not None:
                        st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
                                 caption=selected, use_container_width=True)
            else:
                st.warning("No sample images found.")

    with col2:
        st.subheader("🤖 Detection Results")
        if image_array is not None:
            if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
                detections = []
                annotated = image_array.copy()

                with st.spinner("Running YOLOv8..."):
                    try:
                        from ultralytics import YOLO
                        model = YOLO(model_name)
                        detections = run_detection(image_array, model)
                        annotated = draw_boxes(image_array.copy(), detections)
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 caption="Annotated Output", use_container_width=True)
                        _, buf = cv2.imencode(".jpg", annotated)
                        st.download_button("⬇ Download Annotated Image",
                                           data=buf.tobytes(),
                                           file_name="annotated_output.jpg",
                                           mime="image/jpeg")
                    except ImportError:
                        st.error("Run: pip install ultralytics")

                st.divider()
                st.subheader("📊 Analysis")

                if detections:
                    with st.spinner("Running agent pipeline..."):
                        from vision_pipeline.preprocess import preprocess_image
                        from agents.orchestrator import Orchestrator
                        report = Orchestrator().run([preprocess_image(image_array)])
                        report_dict = report.to_dict()

                    s = report_dict["summary"]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Objects", len(detections))
                    m2.metric("Anomalies", s["total_anomalies"])
                    m3.metric("Layout Issues", s["total_layout_suggestions"])
                    m4.metric("Daily Impact", f"${s['estimated_daily_cost_impact_usd']:.0f}")

                    import pandas as pd
                    st.subheader("🔍 Detected Objects")
                    st.dataframe(pd.DataFrame([
                        {"Label": d["label"], "Class": d["original"], "Confidence": f"{d['confidence']:.1%}"}
                        for d in detections
                    ]), use_container_width=True, hide_index=True)

                    if report_dict["anomalies"]:
                        st.subheader("⚠️ Anomalies")
                        for a in report_dict["anomalies"]:
                            icon = "🔴" if a["severity"] == "critical" else "🟡"
                            with st.expander(f"{icon} {a['type']} — {a['severity'].upper()}"):
                                st.write(a["description"])
                    else:
                        st.success("✅ No anomalies detected")

                    if report_dict["layout_suggestions"]:
                        st.subheader("📐 Layout Suggestions")
                        for ls in report_dict["layout_suggestions"]:
                            icon = "🔴" if ls["priority"] == "high" else "🟡"
                            with st.expander(f"{icon} {ls['category']}"):
                                st.write(ls["description"])
                                st.metric("Est. Saving", f"{ls['estimated_saving_pct']}%")

                    st.download_button("⬇ Download Report (JSON)",
                                       data=json.dumps(report_dict, indent=2),
                                       file_name="warehouse_report.json",
                                       mime="application/json")
                else:
                    st.info("No objects detected. Try lowering confidence or switching to yolov8s.pt")
        else:
            st.info("👈 Upload an image or select a sample to get started.")

# ════════════════════════════════════════════════════════════════
# VIDEO MODE
# ════════════════════════════════════════════════════════════════
else:
    st.subheader("🎥 Video Processing")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Upload a warehouse video**")
        video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

        if video_file:
            st.video(video_file)
            st.caption(f"File: {video_file.name} | Size: {video_file.size / 1024 / 1024:.1f} MB")

    with col2:
        st.markdown("**Processing Controls**")

        if video_file:
            st.info(f"⚙️ Model: `{model_name}` | Confidence: `{conf_threshold}` | Frame skip: every `{frame_skip}` frames")

            if st.button("▶ Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Initialising...")
                status       = st.empty()
                preview      = st.empty()
                metrics_box  = st.empty()

                try:
                    from ultralytics import YOLO
                    model = YOLO(model_name)

                    # Save upload to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    cap = cv2.VideoCapture(tmp_path)
                    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
                    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Output video
                    out_dir = Path("output/video")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"annotated_{video_file.name}"
                    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

                    frame_num        = 0
                    total_detections = 0
                    last_annotated   = None

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_num += 1
                        pct = frame_num / total_frames if total_frames > 0 else 0
                        progress_bar.progress(min(pct, 1.0),
                                              text=f"Processing frame {frame_num}/{total_frames}...")

                        if frame_num % frame_skip == 0:
                            detections = run_detection(frame, model)
                            total_detections += len(detections)
                            annotated = draw_boxes(frame.copy(), detections)

                            # Stats overlay
                            overlay = f"Frame {frame_num} | Detections: {len(detections)}"
                            cv2.putText(annotated, overlay, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(annotated, overlay, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

                            writer.write(annotated)
                            last_annotated = annotated

                            # Show preview every 15 frames
                            if frame_num % 15 == 0 and last_annotated is not None:
                                preview.image(
                                    cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB),
                                    caption=f"Live preview — Frame {frame_num}",
                                    use_container_width=True,
                                )
                                metrics_box.markdown(
                                    f"**Frames processed:** {frame_num} / {total_frames} &nbsp;|&nbsp; "
                                    f"**Total detections:** {total_detections}"
                                )
                        else:
                            writer.write(frame)

                    cap.release()
                    writer.release()
                    progress_bar.progress(1.0, text="✅ Complete!")

                    st.success(f"🎉 Video processed! {frame_num} frames | {total_detections} total detections")

                    # Download button
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "⬇ Download Annotated Video",
                            data=f.read(),
                            file_name=f"annotated_{video_file.name}",
                            mime="video/mp4",
                        )

                    # Show final frame
                    if last_annotated is not None:
                        st.image(cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB),
                                 caption="Final annotated frame", use_container_width=True)

                    # Cloud export
                    if cloud_export != "None":
                        if st.button(f"☁ Export video to {cloud_export}"):
                            if cloud_export == "Google Cloud Storage":
                                from cloud_infra.setup_gcs import upload_image
                                upload_image(out_path, gcs_folder="output/video/")
                            else:
                                from cloud_infra.setup_aws import upload_image
                                upload_image(out_path, s3_folder="output/video/")
                            st.success(f"Uploaded to {cloud_export}!")

                except ImportError:
                    st.error("Run: pip install ultralytics")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("👈 Upload a video file to get started.\n\nSupported formats: MP4, AVI, MOV, MKV")
            st.markdown("""
**Tips for best results:**
- Videos with people or vehicles work best
- Keep videos under 2 minutes for faster processing
- Use `yolov8s.pt` for better accuracy
- Lower frame skip = smoother but slower
            """)
