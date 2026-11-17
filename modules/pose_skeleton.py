# modules/pose_skeleton.py
import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList


class PoseSkeleton:
    """
    Public API:
      - apply(frame_bgr) -> frame_bgr
    """

    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def close(self):
        if hasattr(self.pose, "close"):
            self.pose.close()

    def apply(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if res.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame_bgr,
                res.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame_bgr



class MultiPoseSkeleton:
    """Draw all poses returned by a MediaPipe model in video mode."""

    def __init__(self, model_path):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=2,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self):
        self.landmarker.close()

    def detect(self, frame_bgr, timestamp_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(image, timestamp_ms).pose_landmarks

    def face_regions(self, frame_bgr, timestamp_ms):
        height, width = frame_bgr.shape[:2]
        for pose in self.detect(frame_bgr, timestamp_ms):
            nose = pose[0]
            if nose.visibility is None or nose.visibility < 0.5:
                continue
            left_shoulder, right_shoulder = pose[11], pose[12]
            head_span = max(
                abs(pose[7].x - pose[8].x) * width * 3.5,
                abs(left_shoulder.x - right_shoulder.x) * width,
                48,
            )
            center_x, center_y = nose.x * width, nose.y * height
            left = max(0, int(center_x - head_span / 2))
            top = max(0, int(center_y - head_span * 0.60))
            right = min(width, int(center_x + head_span / 2))
            bottom = min(height, int(center_y + head_span * 0.40))
            if right > left and bottom > top:
                yield left, top, right, bottom

    def apply(self, frame_bgr, timestamp_ms):
        for pose in self.detect(frame_bgr, timestamp_ms):
            landmarks = NormalizedLandmarkList(landmark=[
                NormalizedLandmark(
                    x=point.x, y=point.y, z=point.z,
                    visibility=point.visibility, presence=point.presence,
                )
                for point in pose
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                frame_bgr, landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame_bgr
