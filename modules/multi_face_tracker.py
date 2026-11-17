import cv2
import mediapipe as mp
import itertools
import numpy as np

# =========================
# Utilities (internal)
# =========================
def _create_kcf():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("opencv-contrib-python required")

def _clamp(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return int(x), int(y), int(w), int(h)

def _shrink(x, y, w, h, r):
    cx, cy = x + w*0.5, y + h*0.5
    nw, nh = int(w*(1-r)), int(h*(1-r))
    return int(cx-nw*0.5), int(cy-nh*0.5), nw, nh

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax,bx), max(ay,by)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    return inter / (aw*ah + bw*bh - inter + 1e-9)

def _same_face(first, second):
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    dx = abs((x1 + w1 / 2) - (x2 + w2 / 2))
    dy = abs((y1 + h1 / 2) - (y2 + h2 / 2))
    return dx < max(w1, w2) * 0.65 and dy < max(h1, h2) * 0.65


# =========================
# Internal Track Object
# =========================
class _Track:
    def __init__(self, tid, frame, bbox):
        self.id = tid
        self.tracker = _create_kcf()
        self.tracker.init(frame, list(map(int, bbox)))
        self.bbox = bbox
        self.missed = 0

    def update(self, frame):
        ok, b = self.tracker.update(frame)
        if ok:
            self.bbox = tuple(map(int, b))
            self.missed = 0
        else:
            self.missed += 1
        return ok

    def reinit(self, frame, bbox):
        self.tracker = _create_kcf()
        self.tracker.init(frame, list(map(int, bbox)))
        self.bbox = bbox
        self.missed = 0

# =========================
# Public API
# =========================
class MultiFaceTracker:
    """
    Multi-face ID tracker.
    Public API: update(frame) -> list of {id, bbox}
    """

    def __init__(
        self,
        detect_interval=25,
        scale_w=720,
        shrink_ratio=0.30,
        iou_match=0.25,
        max_missed=20,
        min_confidence=0.30,
        use_face_regions=False,
        detect_on_first_frame=False,
    ):
        self.detect_interval = detect_interval
        self.scale_w = scale_w
        self.shrink_ratio = shrink_ratio
        self.iou_match = iou_match
        self.max_missed = max_missed
        self.detect_on_first_frame = detect_on_first_frame

        self.tracks = []
        self.frame_count = 0
        self.id_gen = itertools.count(1)

        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_confidence
        )

        self.region_detector = (
            mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_confidence)
            if use_face_regions else None
        )

    def close(self):
        self.detector.close()
        if self.region_detector is not None:
            self.region_detector.close()

    def _detection_boxes(self, detections, width, height, offset_x=0, offset_y=0):
        for detection in detections or []:
            box = detection.location_data.relative_bounding_box
            x, y, w, h = _clamp(
                int(box.xmin * width), int(box.ymin * height),
                int(box.width * width), int(box.height * height), width, height,
            )
            x, y, w, h = _shrink(x, y, w, h, self.shrink_ratio)
            if w > 0 and h > 0:
                yield x + offset_x, y + offset_y, w, h

    # =====================
    # Public method
    # =====================
    def update(self, frame, face_regions=None):
        """
        Args:
            frame (BGR)
        Returns:
            list of dict:
            [
              {"id": int, "bbox": (x,y,w,h)},
              ...
            ]
        """
        self.frame_count += 1
        H, W = frame.shape[:2]

        # ---- resize for speed ----
        if W > self.scale_w:
            s = self.scale_w / W
            small = cv2.resize(frame, (self.scale_w, int(H*s)))
        else:
            s = 1.0
            small = frame

        Hs, Ws = small.shape[:2]

        # ---- tracker update ----
        alive = []
        for t in self.tracks:
            if t.update(small) or t.missed < self.max_missed:
                alive.append(t)
        self.tracks = alive

        # ---- detection bridge ----
        first_detection = self.detect_on_first_frame and self.frame_count == 1
        if first_detection or self.frame_count % self.detect_interval == 0:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            detections = self.detector.process(rgb).detections
            full_frame_boxes = list(self._detection_boxes(detections, Ws, Hs))
            dets = full_frame_boxes
            if self.region_detector is not None and face_regions is not None:
                dets = []
                for left, top, right, bottom in face_regions():
                    left, top, right, bottom = (int(v * s) for v in (left, top, right, bottom))
                    crop = rgb[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    resized = cv2.resize(crop, (256, 256))
                    region_faces = self.region_detector.process(resized).detections
                    if region_faces:
                        best = max(region_faces, key=lambda detection: detection.score[0])
                        dets.extend(self._detection_boxes([best], right-left, bottom-top, left, top))

                unique = []
                for box in [*dets, *full_frame_boxes]:
                    if not any(_same_face(box, existing) for existing in unique):
                        unique.append(box)
                    if len(unique) == 2:
                        break
                dets = unique

            used = set()
            confirmed = []
            for t in self.tracks:
                best_iou, best_j = 0, -1
                for j, db in enumerate(dets):
                    if j in used: continue
                    sc = _iou(t.bbox, db)
                    if self.region_detector is not None and _same_face(t.bbox, db):
                        sc = max(sc, self.iou_match)
                    if sc > best_iou:
                        best_iou, best_j = sc, j
                if best_j >= 0 and best_iou >= self.iou_match:
                    t.reinit(small, dets[best_j])
                    used.add(best_j)
                    confirmed.append(t)

            for j, db in enumerate(dets):
                if j not in used:
                    track = _Track(next(self.id_gen), small, db)
                    self.tracks.append(track)
                    confirmed.append(track)

            if self.region_detector is not None:
                for track in self.tracks:
                    if len(confirmed) >= 2:
                        break
                    if track not in confirmed and not any(
                        _same_face(track.bbox, known.bbox) for known in confirmed
                    ):
                        confirmed.append(track)
                self.tracks = confirmed

        # ---- output original coords ----
        out = []
        for t in self.tracks:
            x,y,w,h = t.bbox
            x,y,w,h = int(x/s), int(y/s), int(w/s), int(h/s)
            x,y,w,h = _clamp(x,y,w,h,W,H)
            out.append({"id": t.id, "bbox": (x,y,w,h)})

        return out
