



# 🎥 Smart Surveillance System

   This project is a production-grade **AI-powered surveillance system** that processes live video (webcam or file) to:

- Detect and track multiple people across frames with stable IDs
- Alert when someone has been in frame too long (**loitering detection**)
- Alert when someone enters a defined restricted zone (**zone intrusion detection**)
- Log all events with timestamps to file
- Save screenshots as visual evidence on alert

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Project Overview

This project is a production-grade **AI-powered surveillance system** that processes live video (webcam or file) to:

- Detect and track multiple people across frames with stable IDs
- Alert when someone has been in frame too long (**loitering detection**)
- Alert when someone enters a defined restricted zone (**zone intrusion detection**)
- Log all events with timestamps to file
- Save screenshots as visual evidence on alert

Built to run on a **normal laptop CPU** — no GPU required.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **YOLOv8n Detection** | Lightweight nano model, ~30 FPS on modern CPU |
| 🏷️ **Centroid Tracking** | Stable person IDs across frames without GPU |
| ⏱️ **Loitering Detection** | Configurable time threshold; alerts + logs per person |
| 🚧 **Restricted Zones** | Draw polygon zones; alerts on entry with screenshot |
| 📺 **HUD Overlay** | FPS, person count, live alerts rendered on video |
| 📝 **File Logging** | All alerts saved to `logs/alerts_<timestamp>.log` |
| 📸 **Screenshots** | Auto-saved to `screenshots/` on each alert |
| 🌐 **Streamlit UI** | Browser-based demo with live feed and alert panel |


---

## 🛠 Tech Stack

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** — State-of-the-art real-time object detection
- **[OpenCV](https://opencv.org/)** — Video capture, frame processing, drawing
- **[SciPy](https://scipy.org/)** — Distance matrix for centroid matching
- **[NumPy](https://numpy.org/)** — Array operations
- **[Streamlit](https://streamlit.io/)** — Web interface for demo
- **Python `logging`** — Structured alert logging to file + console

---

## 📁 Project Structure

```
smart_surveillance/
├── main.py          # Core pipeline: YOLO → track → analyze → display
├── tracker.py       # CentroidTracker: assigns stable IDs to detections
├── zones.py         # Zone definitions, polygon test, intrusion alerts
├── utils.py         # Logging, screenshots, HUD drawing, FPS counter
├── app.py           # Streamlit web interface
├── requirements.txt # Python dependencies
├── logs/            # Auto-created: timestamped alert log files
└── screenshots/     # Auto-created: saved frames on each alert
```

---

## 🚀 Setup & Installation

### 1. Clone / download the project
```bash
git clone https://github.com/yourname/smart-surveillance.git
cd smart-surveillance
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> YOLOv8 will auto-download `yolov8n.pt` (~6 MB) on first run.

---

## ▶️ How to Run

### Option A — OpenCV Window (best performance)
```bash
# Webcam
python main.py

# Video file
python main.py --source path/to/video.mp4

# Custom loitering threshold (15 seconds)
python main.py --loiter-threshold 15

# Save annotated output to output.mp4
python main.py --save-output

# All options
python main.py --source 0 --loiter-threshold 8 --save-output
```

### Option B — Streamlit Web UI (demo-friendly)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit the OpenCV window |
| Sidebar sliders | Adjust thresholds live (Streamlit only) |

---

## ⚙️ Configuration

Key constants in `main.py` you can tune:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DEFAULT_LOITER_THRESHOLD` | `8` sec | When to fire loitering alert |
| `YOLO_CONFIDENCE` | `0.45` | Detection sensitivity (lower = more detections) |
| `FRAME_SKIP` | `2` | Run YOLO every N frames (higher = faster) |
| `max_disappeared` | `20` frames | How long to keep a lost track alive |
| `max_distance` | `80` px | Max centroid shift to still match same person |

To change the **restricted zone shape**, edit `zones.py → get_default_zones()` and adjust the `points` list.

---

## 📊 Performance Tips (CPU Optimization)

1. **Use `yolov8n.pt`** — nano model, fastest inference (~6ms/frame on modern CPU)
2. **Increase `FRAME_SKIP`** to 3–4 — run YOLO less often, use cached boxes between runs
3. **Resize input** — passing `imgsz=416` instead of `640` is ~40% faster with minor accuracy loss
4. **Reduce resolution** — `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` speeds up capture
5. **Disable saving** — turn off `--save-output` if you don't need the output file
6. **Close other apps** — YOLO + OpenCV benefit from available RAM and CPU cache

Expected FPS (approximate, laptop CPU):
| Hardware | FPS with frame_skip=2 |
|---|---|
| Modern i7 / M-series | 20–30 FPS |
| Mid-range i5 | 12–20 FPS |
| Older hardware | 6–12 FPS |

---

## 🔧 Extending the System

- **Deep SORT tracking**: Replace `CentroidTracker` in `tracker.py` with Deep SORT for better re-identification across occlusions. Install with `pip install deep-sort-realtime`.
- **More zones**: Add more `zm.add_zone(...)` calls in `zones.py`.
- **Email/Telegram alerts**: Add to `log_alert()` in `utils.py`.
- **Multiple cameras**: Run multiple `main.py` instances with different `--source` values.

---

## 📄 License

MIT License — free to use, modify, and include in your portfolio.

---

## 👤 Author

Built by Ahmed Aboelregal as a portfolio project demonstrating production-level AI & Computer Vision engineering.

### Core Skills Demonstrated:
- Real-time Computer Vision
- Multi-Object Tracking
- Event Detection
- Alert Systems
- Deep Learning Deployment
- Python Backend Engineering
- MLOps & Production Pipelines

### AI Domains:
- Computer Vision
- NLP
- LLM Applications
- Machine Learning
- Deep Learning
