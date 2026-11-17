import cv2
import numpy as np

# =============================
# Feature detectors (init once)
# =============================
orb = cv2.ORB_create(nfeatures=300)
sift = None
if hasattr(cv2, "SIFT_create"):
    sift = cv2.SIFT_create()

# =============================
# Runtime cache (for throttling)
# =============================
_last_sift_frame = None
_last_sift_result = None


# =============================
# apply feature overlay
# =============================
def apply(frame, mask=None, mode="ORB", frame_idx=0):
    """
    frame     : BGR image
    mask      : binary mask or None
    mode      : "ORB" or "SIFT"
    frame_idx : current frame index (for throttling)
    """

    out = frame.copy()

    if mode == "ORB":

        # 每 2 幀才算一次 ORB
        if frame_idx % 5 != 0:
            return frame.copy()
        kps, des = orb.detectAndCompute(frame, mask)
        # kps = sorted(kps, key=lambda x: x.response, reverse=True)[:300]
        out = frame.copy()
        cv2.drawKeypoints(
            out, kps, out,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        return out

    elif mode == "SIFT":
        if sift is None:
            # OpenCV without contrib
            return out
        
        global _last_sift_frame, _last_sift_result
        
        # if frame_idx % 30 != 0 and _last_sift_result is not None:
        #     return _last_sift_result.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kps, des = sift.detectAndCompute(gray, mask)

        sift_out = frame.copy()
        cv2.drawKeypoints(
            sift_out, kps, sift_out,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        _last_sift_result = sift_out.copy()
        return sift_out

    else:
        return out
