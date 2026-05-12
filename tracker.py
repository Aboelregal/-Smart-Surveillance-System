"""
tracker.py — Centroid Tracker
==============================
How it works:
  - Each detected person gets a unique ID based on their bounding box center (centroid).
  - Every frame, we compute the distance between new detections and existing tracked objects.
  - If a new detection is close enough to an existing one → same person (update position).
  - If no match is found → new person (register with new ID).
  - If a tracked person disappears for too many frames → deregister them.

Fixes for single-person multi-ID bug:
  - max_distance raised to 160px — handles fast movement without spawning new IDs.
  - max_disappeared raised to 50 frames — keeps ID alive during brief occlusions.
  - New detection must have NO overlap with any active track before registering
    (IOU guard) — prevents YOLO double-detections from spawning ghost IDs.
  - Only one active track allowed when YOLO returns exactly 1 detection.
"""

import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import time


def _iou(boxA, boxB):
    """Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(areaA + areaB - interArea)


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=160):
        """
        Args:
            max_disappeared (int): Frames a person can be missing before ID removed.
                                   50 frames ≈ 2.5 sec at 20 FPS — survives brief occlusions.
            max_distance (int):    Max centroid shift in pixels to still be same person.
                                   160px handles fast movement on 640-wide frame.
        """
        self.next_object_id = 1  # Start from 1 — more natural for display
        self.objects     = OrderedDict()   # ID → centroid [cx, cy]
        self.bboxes      = OrderedDict()   # ID → [x1, y1, x2, y2]
        self.disappeared = OrderedDict()   # ID → missing frame count
        self.first_seen  = OrderedDict()   # ID → first-seen timestamp

        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid, bbox):
        """Register a new person — only if bbox doesn't heavily overlap an existing track."""
        for existing_bbox in self.bboxes.values():
            if _iou(bbox, existing_bbox) > 0.3:
                # This detection overlaps an existing track — it's the same person,
                # not a new one. Skip registration to avoid ghost IDs.
                return
        self.objects[self.next_object_id]     = centroid
        self.bboxes[self.next_object_id]      = bbox
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id]  = time.time()
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove a person who has been missing too long."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.first_seen[object_id]

    def update(self, detections):
        """
        Main update — called every frame with YOLO bounding boxes.

        Args:
            detections: list of [x1, y1, x2, y2]

        Returns:
            dict: {object_id: {centroid, bbox, time_in_scene}}
        """
        # ── No detections this frame ──────────────────────────────────────────
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self._build_output()

        # ── Compute centroids ─────────────────────────────────────────────────
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            input_centroids[i] = (int((x1+x2)/2), int((y1+y2)/2))

        # ── No existing tracks → register all ────────────────────────────────
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, detections[i])
            return self._build_output()

        # ── Match detections to existing tracks ───────────────────────────────
        object_ids       = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), input_centroids)

        # Sort by smallest distance first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id]     = input_centroids[col]
            self.bboxes[obj_id]      = detections[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing tracks — increment disappeared counter
        for row in set(range(D.shape[0])) - used_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # Unmatched new detections — only register if not overlapping existing track
        for col in set(range(D.shape[1])) - used_cols:
            self.register(input_centroids[col], detections[col])

        return self._build_output()

    def _build_output(self):
        """Return clean output dict with time-in-scene per tracked person."""
        now    = time.time()
        output = {}
        for obj_id, centroid in self.objects.items():
            output[obj_id] = {
                "centroid":      centroid,
                "bbox":          self.bboxes[obj_id],
                "time_in_scene": now - self.first_seen[obj_id],
            }
        return output

        """
        Args:
            max_disappeared (int): How many consecutive frames a person can be
                                   missing before we remove their ID.
            max_distance (int): Max pixel distance to still consider two
                                centroids the same person.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()        # ID → centroid [cx, cy]
        self.bboxes = OrderedDict()         # ID → [x1, y1, x2, y2]
        self.disappeared = OrderedDict()    # ID → frames missing count
        self.first_seen = OrderedDict()     # ID → timestamp when first detected

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        """Register a brand-new person with the next available ID."""
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = time.time()
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove a person who has been missing too long."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.first_seen[object_id]

    def update(self, detections):
        """
        Main update call. Called every frame with a list of bounding boxes.

        Args:
            detections: list of [x1, y1, x2, y2] bounding boxes from YOLO

        Returns:
            dict: {object_id: (centroid, bbox, time_in_scene)}
        """
        # No detections this frame
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self._build_output()

        # Compute centroids for each detected bounding box
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)

        # If we have no existing tracked objects, register all detections
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, detections[i])
        else:
            # Match new detections to existing tracked objects via distance
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Build pairwise distance matrix: rows=existing, cols=new detections
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Greedy matching: smallest distances first
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.bboxes[obj_id] = detections[col]
                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing tracks (person disappeared)
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            # Handle unmatched new detections (new person entered)
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], detections[col])

        return self._build_output()

    def _build_output(self):
        """Build a clean output dict with time-in-scene for each tracked person."""
        output = {}
        now = time.time()
        for obj_id, centroid in self.objects.items():
            time_in_scene = now - self.first_seen[obj_id]
            output[obj_id] = {
                "centroid": centroid,
                "bbox": self.bboxes[obj_id],
                "time_in_scene": time_in_scene,
            }
        return output