"""
utils.py — Utilities: Logging, Alerts, Overlays
=================================================
"""

import cv2
import os
import time
import logging
from datetime import datetime


# ── Logging setup ───────────────────────────────────────────────────────────
LOG_DIR = "logs"
SCREENSHOT_DIR = "screenshots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SurveillanceSystem")


# ── Alert + screenshot ───────────────────────────────────────────────────────
_last_alert_times = {}
ALERT_COOLDOWN_SEC = 2


def log_alert(message: str, frame=None, alert_type: str = "general"):
    """
    Log an alert message and optionally queue a screenshot.
    Screenshot is queued (not saved immediately) so drawings are included.
    """
    now = time.time()
    if now - _last_alert_times.get(message, 0) < ALERT_COOLDOWN_SEC:
        return

    _last_alert_times[message] = now
    logger.warning(f"[ALERT] {message}")

    if frame is not None:
        queue_screenshot(alert_type)

    try:
        play_beep()
    except Exception:
        pass


def save_screenshot(frame, tag: str = "alert"):
    """Save a timestamped screenshot to the screenshots/ directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(SCREENSHOT_DIR, f"{tag}_{ts}.jpg")
    cv2.imwrite(filename, frame.copy())
    logger.info(f"[Screenshot saved] {filename}")


# ── Deferred screenshot queue ────────────────────────────────────────────────
_screenshot_queue = []


def queue_screenshot(tag: str = "alert"):
    """Queue a screenshot — actual save happens after drawings via flush_screenshots()."""
    _screenshot_queue.append(tag)


def flush_screenshots(frame):
    """Call AFTER all drawings are applied. Saves all queued screenshots."""
    global _screenshot_queue
    for tag in _screenshot_queue:
        save_screenshot(frame, tag)
    _screenshot_queue.clear()


def play_beep():
    """Platform-aware audio beep. Fails silently if unavailable."""
    import sys
    if sys.platform == "win32":
        import winsound
        winsound.Beep(1000, 300)
    elif sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Ping.aiff &")
    else:
        print("\a", end="", flush=True)


# ── HUD overlay ─────────────────────────────────────────────────────────────
def draw_hud(frame, fps: float, person_count: int, active_alerts: list):
    h, w = frame.shape[:2]

    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (260, 65), (20, 20, 20), -1)
    cv2.addWeighted(banner, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
    cv2.putText(frame, f"People: {person_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    for i, alert in enumerate(active_alerts[-4:]):
        y = h - 20 - i * 28
        cv2.putText(frame, f"!! {alert}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 255), 2)


def draw_person(frame, obj_id: int, bbox, time_in_scene: float,
                is_loitering: bool = False, zone_names: list = None):
    x1, y1, x2, y2 = [int(v) for v in bbox]

    if is_loitering or zone_names:
        color = (0, 0, 255)
    else:
        color = (0, 200, 50)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{obj_id}  {time_in_scene:.0f}s"
    if is_loitering:
        label += "  [LOITERING]"
    if zone_names:
        label += f"  [ZONE:{','.join(zone_names)}]"

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = max(y1 - 10, label_size[1] + 5)

    cv2.rectangle(frame,
                  (x1, label_y - label_size[1] - 5),
                  (x1 + label_size[0] + 4, label_y + 2),
                  color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ── FPS calculator ───────────────────────────────────────────────────────────
class FPSCounter:
    """Rolling average FPS counter over the last N frames."""

    def __init__(self, window=30):
        self.window = window
        self._times = []

    def tick(self):
        self._times.append(time.time())
        if len(self._times) > self.window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0