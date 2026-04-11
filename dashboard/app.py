"""
Warehouse Visual Intelligence - Streamlit Dashboard
Run with: streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path
import numpy as np

import streamlit as st

# Allow imports from project root
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Warehouse Visual Intelligence",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Warehouse Visual Intelligence System")
st.caption("Multi-agent AI for warehouse monitoring, safety & cost optimisation")

# ─── Sidebar Controls ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    use_cloud = st.toggle("Use Google Cloud Vision API", value=False)
    confidence_threshold = st.slider("Detection confidence threshold", 0.1, 1.0, 0.4)
    st.divider()
    st.info("Upload an image or select a sample to run the pipeline.")

# ─── Image Input ────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Input Image")
    uploaded_file = st.file_uploader("Upload warehouse image", type=["jpg", "jpeg", "png"])

    use_sample = st.checkbox("Use sample image instead")

    image_array = None

    if uploaded_file:
        import cv2
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

    elif use_sample:
        sample_path = Path("data/sample_images")
        samples = list(sample_path.glob("*.jpg")) + list(sample_path.glob("*.png"))
        if samples:
            selected = st.selectbox("Choose sample", [p.name for p in samples])
            import cv2
            image_array = cv2.imread(str(sample_path / selected))
            st.image(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), caption=selected, use_column_width=True)
        else:
            st.warning("No sample images found in data/sample_images/")

# ─── Run Pipeline ───────────────────────────────────────────────────
with col2:
    st.subheader("🤖 Agent Results")

    if image_array is not None and st.button("▶ Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running multi-agent pipeline..."):
            from vision_pipeline.preprocess import preprocess_image
            from agents.orchestrator import Orchestrator

            processed = preprocess_image(image_array)
            orch = Orchestrator()
            report = orch.run([processed])
            report_dict = report.to_dict()

        st.success("Pipeline complete!")

        # Summary metrics
        s = report_dict["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Objects Detected", s["total_detections"])
        m2.metric("Anomalies", s["total_anomalies"])
        m3.metric("Layout Issues", s["total_layout_suggestions"])
        m4.metric("Daily Cost Impact", f"${s['estimated_daily_cost_impact_usd']:.0f}")

        st.divider()

        # Anomalies
        if report_dict["anomalies"]:
            st.subheader("⚠️ Anomalies")
            for a in report_dict["anomalies"]:
                severity_icon = "🔴" if a["severity"] == "critical" else "🟡"
                with st.expander(f"{severity_icon} {a['type']} — {a['severity'].upper()}"):
                    st.write(a["description"])
                    st.caption(f"Location: {a['location']}")
        else:
            st.success("✅ No anomalies detected")

        # Layout suggestions
        if report_dict["layout_suggestions"]:
            st.subheader("📐 Layout Suggestions")
            for s in report_dict["layout_suggestions"]:
                priority_icon = "🔴" if s["priority"] == "high" else "🟡"
                with st.expander(f"{priority_icon} {s['category']} — {s['priority'].upper()} priority"):
                    st.write(s["description"])
                    st.metric("Estimated Saving", f"{s['estimated_saving_pct']}%")

        # JSON download
        st.download_button(
            "⬇ Download Report JSON",
            data=json.dumps(report_dict, indent=2),
            file_name="warehouse_report.json",
            mime="application/json",
        )
    elif image_array is None:
        st.info("Upload an image or select a sample to get started.")
