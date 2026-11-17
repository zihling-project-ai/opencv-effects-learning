# modules/main_face_mesh.py
import cv2
import mediapipe as mp


class MainFaceMesh:
    """
    Main face selector + FaceMesh drawer
    Public API:
      - process(frame_bgr) -> dict { "bbox": (x,y,w,h) or None, "landmarks": face_lms or None }
      - draw(frame_bgr, bbox, landmarks, draw_bbox=True, draw_mesh=True) -> frame_bgr
      - apply(frame_bgr, draw_bbox=True, draw_mesh=True) -> (frame_bgr, info)
    """

    def __init__(
        self,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        bbox_expand=0.05
    ):
        self.bbox_expand = float(bbox_expand)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def close(self):
        # 有些版本支援 close，保險寫法
        if hasattr(self.face_mesh, "close"):
            self.face_mesh.close()

    def process(self, frame_bgr):
        """
        找主臉：用 landmarks 的 bbox 面積最大者當主臉
        回傳：
          bbox: (x,y,w,h) in pixel coords（原圖座標）
          landmarks: MediaPipe face landmarks（供畫 mesh 或進一步分析）
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"bbox": None, "landmarks": None}

        best_landmarks = None
        best_area = -1.0
        best_bbox = None  # (xmin,ymin,xmax,ymax) in normalized coords

        for face_lms in results.multi_face_landmarks:
            xs = [lm.x for lm in face_lms.landmark]
            ys = [lm.y for lm in face_lms.landmark]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best_landmarks = face_lms
                best_bbox = (xmin, ymin, xmax, ymax)

        xmin, ymin, xmax, ymax = best_bbox

        # bbox expand（避免太貼臉邊）
        bw = xmax - xmin
        bh = ymax - ymin
        e = self.bbox_expand

        xmin -= bw * e
        ymin -= bh * e
        xmax += bw * e
        ymax += bh * e

        # clamp + to pixel bbox
        x1 = int(max(0, xmin * w))
        y1 = int(max(0, ymin * h))
        x2 = int(min(w - 1, xmax * w))
        y2 = int(min(h - 1, ymax * h))

        if x2 <= x1 or y2 <= y1:
            return {"bbox": None, "landmarks": best_landmarks}

        bbox = (x1, y1, x2 - x1, y2 - y1)
        return {"bbox": bbox, "landmarks": best_landmarks}
    
    def _apply_soft_glow(self, roi):
        """
        柔焦濾鏡（自然、不會塑膠臉）
        """
        blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=8, sigmaY=8)
        soft = cv2.addWeighted(roi, 1.0, blur, 0.25, 0)
        return soft

    def draw(self, frame_bgr, bbox, landmarks, draw_bbox=True, draw_mesh=True):
        """
        在 frame 上畫 bbox / mesh（可選）
        """
        if bbox is None and landmarks is None:
            return frame_bgr

        if draw_bbox and bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                "Main Face (Face Mesh)",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if draw_mesh and landmarks is not None:
            self.mp_draw.draw_landmarks(
                image=frame_bgr,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()
            )

        return frame_bgr

    def apply(
        self,
        frame_bgr,
        current_sec=None,
        hide_bbox_after=None,
        draw_mesh=True,
        enable_beauty=True
        ):
        """
        Args:
            current_sec (float): 目前影片秒數（由 main 傳入）
            hide_bbox_after (float): 幾秒後不畫 bbox
        """
        info = self.process(frame_bgr)
        bbox = info["bbox"]
        landmarks = info["landmarks"]

        # ===== Beauty / Soft Glow（只吃 ROI）=====
        if enable_beauty and bbox is not None:
            x, y, w, h = bbox
            roi = frame_bgr[y:y+h, x:x+w].copy()

            if roi.size > 0:
                roi = self._apply_soft_glow(roi)
                frame_bgr[y:y+h, x:x+w] = roi

        # ===== 是否顯示 bbox（時間控制）=====
        draw_bbox = True
        if current_sec is not None and hide_bbox_after is not None:
            if current_sec >= hide_bbox_after:
                draw_bbox = False

        # ===== Draw overlay =====
        self.draw(
            frame_bgr,
            bbox=bbox,
            landmarks=landmarks,
            draw_bbox=draw_bbox,
            draw_mesh=draw_mesh
        )

        return frame_bgr, info
