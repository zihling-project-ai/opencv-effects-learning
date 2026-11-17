"""
Edge effect using Canny
input:
    frame (BGR image to Gray, np.ndarray)
output:
    BGR image with edge effect
"""

import cv2
import numpy as np

# =============================
#  Edge function
# =============================
def sobel_edge(gray):
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sobel = cv2.addWeighted(
        cv2.convertScaleAbs(sx), 0.5,
        cv2.convertScaleAbs(sy), 0.5, 0
    )
    return sobel

def laplacian_edge(gray):
    lap = cv2.Laplacian(gray, cv2.CV_16S)
    return cv2.convertScaleAbs(lap)

def canny_edge(gray):
    return cv2.Canny(gray, 80, 160)

def morph_gradient_edge(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

def dog_edge(gray):
    d = 5
    img_D5 = cv2.GaussianBlur(gray, (d, d), 0.5)
    img_D13 = cv2.GaussianBlur(gray, (d, d), 1.3)   
    diff = img_D13 - img_D5
    return diff

# =============================
# apply Edge f(x)
# =============================
def apply(frame, mode):

    """
    mode:
        sobel_laplacian  → left: Sobel / right: Laplacian
        canny_morphgradient → left: Canny / right: Morph Gradient
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "sobel_laplacian":
        left = sobel_edge(gray)
        right = laplacian_edge(gray)
    
    elif mode == "canny_morphgradient":
        left = canny_edge(gray)
        right = morph_gradient_edge(gray)
    
    elif mode == "DOG":
        dog = dog_edge(gray)
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog = dog.astype(np.uint8)
        dog = cv2.cvtColor(dog, cv2.COLOR_GRAY2BGR)
        return dog

    else:
        return frame.copy()

    # Turn to BGR to concat
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    # 垂直二分割
    combined = np.hstack([left, right])

    # resize to original size (Error Proofing)
    combined = cv2.resize(combined, (w, h))

    return combined