"""
main.py — Smart Surveillance System Entry Point
================================================
Pipeline overview (runs every frame):
  1. Capture frame from webcam or video file (OpenCV)
  2. Run YOLOv8n inference → get human bounding boxes
  3. Pass boxes to CentroidTracker → get stable person IDs + time-in-scene
  4. For each tracked person:
       a. Check loitering threshold → alert if exceeded
       b. Check restricted zones   → alert if inside
       c. Draw bounding box, ID, labels on frame
  5. Draw HUD overlay (FPS, count, active alerts)
  6. Show frame (and optionally write to output video)

Usage:
  # Webcam (default)
  python main.py

  # Video file
  python main.py --source path/to/video.mp4

  # Custom loitering threshold (seconds)
  python main.py --loiter-threshold 10

  # Save output video
  python main.py --save-output
"""

import cv2
import argparse
import sys
import logging

from tracker import CentroidTracker
from zones import get_default_zones
from utils import draw_hud, draw_person, log_alert, FPSCounter, flush_screenshots

# Try importing ultralytics (YOLOv8) — give a helpful message if not installed
try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] ultralytics not installed.")
    print("Run: pip install ultralytics\n")
    sys.exit(1)

logger = logging.getLogger("SurveillanceSystem")


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_LOITER_THRESHOLD = 15     # seconds before loitering alert
YOLO_CONFIDENCE = 0.45            # minimum detection confidence (0–1)
YOLO_MODEL = "yolov8n.pt"         # nano model — fast on CPU
PERSON_CLASS_ID = 0               # COCO class 0 = "person"
FRAME_SKIP = 2                    # Run YOLO every N frames (1 = every frame)
                                  # Increase to 3–4 if FPS is too low on your machine


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Surveillance System")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, or path to video file")
    parser.add_argument("--loiter-threshold", type=float, default=DEFAULT_LOITER_THRESHOLD,
                        help="Seconds before loitering alert fires (default: 8)")
    parser.add_argument("--save-output", action="store_true",
                        help="Save annotated output to output.mp4")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no window). Useful for server deployments.")
    return parser.parse_args()


def load_video_source(source_str: str):
    """Open webcam (int) or video file (str)."""
    source = int(source_str) if source_str.isdigit() else source_str
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source_str}")
        sys.exit(1)
    return cap


def main():
    args = parse_args()
    logger.info("=== Smart Surveillance System Starting ===")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    # Warm up the model (first inference is always slower)
    import numpy as np
    model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
    logger.info("Model loaded and warmed up.")

    # ── Video source ──────────────────────────────────────────────────────────
    cap = load_video_source(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video source opened: {frame_w}x{frame_h}")

    # ── Components ────────────────────────────────────────────────────────────
    tracker = CentroidTracker(max_disappeared=50, max_distance=160)
    zone_manager = get_default_zones(frame_w, frame_h)
    fps_counter = FPSCounter(window=30)

    # ── Output video writer (optional) ────────────────────────────────────────
    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 20, (frame_w, frame_h))
        logger.info("Saving output to output.mp4")

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    last_detections = []   # Cache detections between YOLO runs (frame-skip)

    logger.info("Processing started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        # ── Mirror frame immediately so all drawings are on correct side ─────
        frame = cv2.flip(frame, 1)

        frame_idx += 1
        fps_counter.tick()

        # ── YOLOv8 inference (with frame skipping for speed) ──────────────────
        if frame_idx % FRAME_SKIP == 0:
            # Resize for faster inference (keeps aspect ratio via padding)
            results = model(frame, imgsz=640, conf=YOLO_CONFIDENCE,
                            classes=[PERSON_CLASS_ID], verbose=False)

            last_detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                last_detections.append([x1, y1, x2, y2])

        # ── Centroid tracking ─────────────────────────────────────────────────
        tracked = tracker.update(last_detections)

        # ── Per-person analysis ───────────────────────────────────────────────
        active_alerts = []

        for obj_id, info in tracked.items():
            centroid  = info["centroid"]
            bbox      = info["bbox"]
            time_in_scene = info["time_in_scene"]

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Use body centroid for zone detection (works for any zone position)
            check_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Loitering check
            is_loitering = time_in_scene >= args.loiter_threshold
            if is_loitering:
                msg = f"LOITERING | Person #{obj_id} for {time_in_scene:.0f}s"
                active_alerts.append(f"Person #{obj_id} loitering {time_in_scene:.0f}s")
                log_alert(msg, frame=frame, alert_type="loitering")

            # Zone intrusion check
            zone_hits = zone_manager.check_all(obj_id, check_point, frame)
            if zone_hits:
                active_alerts.append(f"ZONE ALERT: Person #{obj_id}!")
                logger.warning(f"[ZONE] Person #{obj_id} center={check_point} → {zone_hits}")

            # Draw person annotation
            draw_person(frame, obj_id, bbox, time_in_scene,
                        is_loitering=is_loitering, zone_names=zone_hits)

            # Draw the check point as a small dot so you can see where detection fires
            cv2.circle(frame, check_point, 5, (0, 255, 255), -1)

        # ── Draw zones + HUD ──────────────────────────────────────────────────
        zone_manager.draw_all(frame)
        draw_hud(frame, fps_counter.fps, len(tracked), active_alerts)

        # ── Save screenshots AFTER all drawings so frames aren't empty ────────
        flush_screenshots(frame)

        # ── Display ───────────────────────────────────────────────────────────
        if not args.no_display:
            win_name = "Smart Surveillance System  [q = quit]"

            # On first frame: create a normal window, resize and center it
            if frame_idx == 1:
                WIN_W, WIN_H = 1152, 648          # ~60% of 1920x1080
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, WIN_W, WIN_H)
                # Center on a 1920x1080 screen — adjust SCREEN_W/H if different
                SCREEN_W, SCREEN_H = 1920, 1080
                cv2.moveWindow(win_name,
                               (SCREEN_W - WIN_W) // 2,
                               (SCREEN_H - WIN_H) // 2)

            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit.")
                break

        # ── Write output ──────────────────────────────────────────────────────
        if writer:
            writer.write(frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("=== System shut down cleanly ===")


if __name__ == "__main__":
    main()