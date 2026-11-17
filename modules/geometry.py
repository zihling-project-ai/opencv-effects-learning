"""
Geometry effect using Resize, Rotate, Flip, 4-in-1
"""

import cv2
import numpy as np
import math

# =============================
# 1. Resize + Rotate
# =============================

def resize_rotate(frame, progess):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # scale
    scale = 0.3 + 0.7 *progess

    # angel: 360 度旋轉
    angle = 360 * progess

    # resize
    resized = cv2.resize(
        frame,
        (int(w * scale), int(h * scale))
    )
    
    # 放回黑底
    canvas = np.zeros_like(frame)
    rh, rw = resized.shape[:2]
    x = cx - rw // 2
    y = cy - rh // 2
    canvas[y:y+rh, x:x+rw] = resized

    # rotate
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (w, h))
    
    return rotated

# =============================
# 2. Flip 4-in-1
# =============================

def flip_4in1(frame):
    h, w = frame.shape[:2]
    half_h, half_w = h // 2, w // 2

    f1 = cv2.resize(cv2.flip(frame, 1), (half_w, half_h))   # horizontal
    f2 = cv2.resize(cv2.flip(frame, 0), (half_w, half_h))   # vertical
    f3 = cv2.resize(cv2.flip(frame, -1), (half_w, half_h))  # diagonal
    f4 = cv2.resize(frame, (half_w, half_h))                # original

    top = np.hstack([f4, f1])
    bottom = np.hstack([f2, f3])

    return np.vstack([top, bottom])


# =============================
#  apply
# =============================

def apply(frame, mode="resize_rotate", progress=0.0):
    """
    mode:
        resize_rotate
        flip_4in1
    """
    if mode == "resize_rotate":
        return resize_rotate(frame, progress)

    elif mode == "flip_4in1":
        return flip_4in1(frame)

    else:
        return frame.copy()