import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Drag a box to blur part of the first video frame.")
    parser.add_argument("--input", required=True, help="Path to your video.")
    args = parser.parse_args()

    video_path = Path(args.input).expanduser()
    if not video_path.is_file():
        parser.error(f"Input video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            parser.error(f"Cannot open video: {video_path}")
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        parser.error(f"Cannot read the requested frame: {video_path}")

    orig = frame.copy()
    display = frame.copy()


    #=============== Mouse state ================
    drawing = False
    x0, y0 = -1, -1
    bbox = None #(x, y, w, h)

    #=============== Mouse callback ================
    def on_mouse(event, x, y, flag, parm):
        nonlocal drawing, x0, y0, bbox, display

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y
            bbox = None
    
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            display = orig.copy()
            cv2.rectangle(display, (x0, y0), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = x, y

            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)

            if w > 0 and h > 0:
                bbox = (x, y, w, h)
                apply_roi_effect()

    # ================== ROI + Mask effect ==================
    def apply_roi_effect():
        nonlocal display

        display = orig.copy()

        if bbox is None:
            return
    
        x, y, w, h = bbox

        #----ROI-----
        roi = orig[y:y+h, x:x+w]

        #---- simple effect: bur ----
        roi_blur = cv2.GaussianBlur(roi, (21, 21), 0)

        #---- put ROI back --------
        display[y:y+h, x:x+w] = roi_blur

        #---- draw bbox (Learning purpose) ---
        cv2.rectangle(display, (x,y), (x+w, y+h), (0, 255, 0), 2)

    # ================== UI ==================
    cv2.namedWindow("PhaseA - Bounding Box Demo")
    cv2.setMouseCallback("PhaseA - Bounding Box Demo", on_mouse)

    print("Drag mouse to draw Bounding Box. Press ESC to exit.")

    # ================== Main loop ==================

    while True:
        cv2.imshow("PhaseA - Bounding Box Demo", display)
        key = cv2.waitKey(20)

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
