"""
app.py — Streamlit Web Interface for Smart Surveillance System
==============================================================
Provides a browser-based demo UI with:
  - Sidebar controls: loitering threshold, confidence, frame-skip
  - Live video feed rendered in the browser (via st.image)
  - Real-time alert log panel
  - Zone visualization toggle

Run with:
    streamlit run app.py

Note: Streamlit updates the displayed frame at ~10–15 FPS in browser.
For true real-time performance, use main.py directly with OpenCV window.
"""

import cv2
import time
import numpy as np
import streamlit as st
from datetime import datetime

# Internal modules
from tracker import CentroidTracker
from zones import get_default_zones
from utils import FPSCounter, draw_person, draw_hud, log_alert

try:
    from ultralytics import YOLO
except ImportError:
    st.error("ultralytics not installed. Run: pip install ultralytics")
    st.stop()


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 Smart Surveillance System")
st.caption("YOLOv8 · Centroid Tracking · Loitering & Zone Detection")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

video_source = st.sidebar.text_input("Video Source", value="0",
    help="'0' = webcam, or paste a path to a .mp4 file")
loiter_threshold = st.sidebar.slider("Loitering Threshold (seconds)", 3, 30, 8)
confidence = st.sidebar.slider("YOLO Confidence", 0.2, 0.9, 0.45, 0.05)
frame_skip = st.sidebar.slider("Frame Skip (higher = faster)", 1, 5, 2,
    help="Run YOLO every N frames. Increase if FPS is low.")
show_zones = st.sidebar.checkbox("Show Zone Overlays", value=True)
save_screenshots = st.sidebar.checkbox("Save Alert Screenshots", value=True)

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶ Start", use_container_width=True)
stop_btn = col2.button("⏹ Stop", use_container_width=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_video, col_alerts = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    st.caption("Live annotated feed")

with col_alerts:
    st.subheader("🚨 Alert Log")
    alert_placeholder = st.empty()

stats_col1, stats_col2, stats_col3 = st.columns(3)
fps_metric = stats_col1.empty()
count_metric = stats_col2.empty()
alert_count_metric = stats_col3.empty()

# ── Session state ─────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "alerts" not in st.session_state:
    st.session_state.alerts = []

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ── Inference loop ────────────────────────────────────────────────────────────
if st.session_state.running:
    @st.cache_resource
    def load_model():
        m = YOLO("yolov8n.pt")
        m(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
        return m

    model = load_model()
    source = int(video_source) if video_source.strip().isdigit() else video_source.strip()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error(f"Cannot open video source: {video_source}")
        st.stop()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = CentroidTracker(max_disappeared=20, max_distance=80)
    zone_manager = get_default_zones(frame_w, frame_h)
    fps_counter = FPSCounter(window=30)

    frame_idx = 0
    last_detections = []
    total_alerts = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or camera disconnected.")
            break

        frame_idx += 1
        fps_counter.tick()

        if frame_idx % frame_skip == 0:
            results = model(frame, imgsz=640, conf=confidence,
                            classes=[0], verbose=False)
            last_detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                last_detections.append([x1, y1, x2, y2])

        tracked = tracker.update(last_detections)
        active_alerts = []

        for obj_id, info in tracked.items():
            centroid = info["centroid"]
            bbox = info["bbox"]
            time_in_scene = info["time_in_scene"]

            is_loitering = time_in_scene >= loiter_threshold
            if is_loitering:
                msg = f"[{datetime.now().strftime('%H:%M:%S')}] Person #{obj_id} loitering {time_in_scene:.0f}s"
                if msg not in st.session_state.alerts:
                    st.session_state.alerts.insert(0, msg)
                    total_alerts += 1
                active_alerts.append(f"Person #{obj_id} loitering {time_in_scene:.0f}s")

            zone_hits = zone_manager.check_all(obj_id, centroid,
                                               frame if save_screenshots else None)
            if zone_hits:
                msg = f"[{datetime.now().strftime('%H:%M:%S')}] Person #{obj_id} in {zone_hits[0]}"
                if msg not in st.session_state.alerts:
                    st.session_state.alerts.insert(0, msg)
                    total_alerts += 1
                active_alerts.append(f"Person #{obj_id} in {zone_hits[0]}")

            draw_person(frame, obj_id, bbox, time_in_scene,
                        is_loitering=is_loitering, zone_names=zone_hits)

        if show_zones:
            zone_manager.draw_all(frame)
        draw_hud(frame, fps_counter.fps, len(tracked), active_alerts)

        # Streamlit expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update stats
        fps_metric.metric("FPS", f"{fps_counter.fps:.1f}")
        count_metric.metric("People", len(tracked))
        alert_count_metric.metric("Total Alerts", total_alerts)

        # Alert log (show last 20)
        alert_text = "\n".join(st.session_state.alerts[:20])
        alert_placeholder.text_area("", value=alert_text, height=400,
                                    label_visibility="collapsed")

        time.sleep(0.01)  # Yield to Streamlit's async runtime

    cap.release()
else:
    video_placeholder.info("👆 Press **▶ Start** to begin surveillance feed.")
    alert_placeholder.text_area("", value="No alerts yet.", height=400,
                                label_visibility="collapsed")
