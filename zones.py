"""
zones.py — Zone / Restricted Area Management
==============================================
How it works:
  - You define one or more polygonal zones by listing their corner points.
  - Each frame, for every tracked person, we check if their centroid falls
    inside any restricted zone using OpenCV's pointPolygonTest.
  - If a person is inside a zone → fire a zone intrusion alert.

Why polygons instead of rectangles?
  - Real-world restricted areas (doorways, stairwells, vault areas) are rarely
    perfect rectangles. Polygons let you trace any shape on screen.
"""

import cv2
import numpy as np
from utils import log_alert


class Zone:
    def __init__(self, name: str, points: list, color=(0, 0, 255), alpha=0.25):
        """
        Args:
            name   : Human-readable zone label, e.g. "Server Room"
            points : List of (x, y) tuples defining the polygon corners.
                     e.g. [(100, 100), (300, 100), (300, 300), (100, 300)]
            color  : BGR color for drawing the zone overlay.
            alpha  : Transparency of the filled zone overlay (0=invisible, 1=solid).
        """
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = color
        self.alpha = alpha
        self._alerted_ids = set()   # Track which person IDs we've already alerted for

    def is_inside(self, point) -> bool:
        """
        Returns True if the given (x, y) point is inside this polygon zone.
        Uses OpenCV's pointPolygonTest (returns >0 if inside, 0 on edge, <0 outside).
        """
        result = cv2.pointPolygonTest(self.points, (float(point[0]), float(point[1])), False)
        return result >= 0

    def draw(self, frame):
        """
        Draw a semi-transparent filled polygon + border on the frame.
        Uses an overlay blend so the video feed is still visible underneath.
        """
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], self.color)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        cv2.polylines(frame, [self.points], isClosed=True, color=self.color, thickness=2)

        # ── Label perfectly centered inside the zone ──────────────────────────
        label      = "Restricted Zone"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 2

        # Compute zone center
        cx = int(self.points[:, 0].mean())
        cy = int(self.points[:, 1].mean())

        # Measure text dimensions
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Text top-left so it ends up centered
        tx = cx - tw // 2
        ty = cy + th // 2

        # Dark background pill for readability
        pad = 5
        cv2.rectangle(frame,
                      (tx - pad, ty - th - pad),
                      (tx + tw + pad, ty + baseline + pad),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty),
                    font, font_scale, (255, 255, 255), thickness)

    def check_intrusion(self, object_id: int, centroid, frame) -> bool:
        """
        Check if a person (by ID + centroid) is inside this zone.
        Only fires the alert ONCE per person entry (not every frame).

        Returns True if an intrusion is happening this frame.
        """
        if self.is_inside(centroid):
            if object_id not in self._alerted_ids:
                self._alerted_ids.add(object_id)
                msg = f"ZONE INTRUSION | Person #{object_id} entered '{self.name}'"
                log_alert(msg, frame=frame, alert_type="zone")
            return True
        else:
            # Person left the zone — reset so we alert again if they re-enter
            self._alerted_ids.discard(object_id)
            return False


class ZoneManager:
    """Manages a collection of Zone objects."""

    def __init__(self):
        self.zones = []

    def add_zone(self, name, points, color=(0, 0, 255)):
        self.zones.append(Zone(name, points, color))

    def draw_all(self, frame):
        for zone in self.zones:
            zone.draw(frame)

    def check_all(self, object_id, centroid, frame) -> list:
        """
        Check all zones for a given person.
        Returns list of zone names the person is currently inside.
        """
        triggered = []
        for zone in self.zones:
            if zone.check_intrusion(object_id, centroid, frame):
                triggered.append(zone.name)
        return triggered


def get_default_zones(frame_width, frame_height):
    """
    Returns a demo ZoneManager with a default restricted zone
    placed in the upper-right quadrant of the frame.
    You can edit these coordinates or add more zones as needed.
    """
    zm = ZoneManager()

    # Restricted zone: right side of frame, full height — easy to walk into
    w, h = frame_width, frame_height
    zm.add_zone(
        name="Restricted Area",
        points=[
            (int(w * 0.65), int(h * 0.05)),
            (int(w * 0.98), int(h * 0.05)),
            (int(w * 0.98), int(h * 0.95)),
            (int(w * 0.65), int(h * 0.95)),
        ],
        color=(0, 0, 220),
    )
    return zm