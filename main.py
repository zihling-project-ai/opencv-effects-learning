import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2

# ================== Effects ==================
from modules.edge import apply as edge_fx
from modules.colormap import apply as color_fx
from modules.geometry import apply as geometry_fx
from modules.features import apply as feature_fx
from modules.multi_face_tracker import MultiFaceTracker
from modules.main_face_mesh import MainFaceMesh
from modules.pose_skeleton import MultiPoseSkeleton, PoseSkeleton


# ================== FX Timeline（新增） ==================
FX_TIMELINE = [
    # {"start": 0,  "end": 10, "fx": "original", "pip": False},
    {"start": 10, "end": 17, "fx": "edge", "mode": "sobel_laplacian",  "pip": True},
    {"start": 17, "end": 23, "fx": "edge", "mode": "canny_morphgradient", "pip": True},
    {"start": 23, "end": 30, "fx": "edge", "mode": "DOG", "pip": True},
    {"start": 30, "end": 35, "fx": "colormap", "pip": False},
    {"start": 35, "end": 42, "fx": "geometry", "mode": "resize_rotate", "pip": False},
    {"start": 42, "end": 50, "fx": "geometry", "mode": "flip_4in1", "pip": False},
    {"start": 50, "end": 60, "fx": "feature", "mode": "ORB", "pip": False},
    {"start": 60, "end": 65, "fx": "feature", "mode": "SIFT", "pip": True},
]

def get_fx_config(sec):
    for seg in FX_TIMELINE:
        if seg["start"] <= sec <= seg["end"]:
            return seg
    return {"fx": "original", "pip": False}

# ================== UI ==================
def draw_transparent_box(img, p1, p2, alpha=0.5):
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_overlay_ui(img, frame_idx, total, fx_name, fps):
    draw_transparent_box(img, (20, 20), (420, 150), alpha=0.55)

    cv2.putText(img, f"Frame : {frame_idx}/{total}",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    cv2.putText(img, f"Effect : {fx_name}",
                (40, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.putText(img, f"FPS : {fps:.1f}",
                (40, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

class VideoProcessingCancelled(Exception):
    """The user cancelled the video from the preview window."""


def render_frames(cap, writer, fps_video, total, out_w, out_h, preview, resources, multi_person_model):
    # Keep the original detector settings for the default mode.
    tracker_options = {}
    if multi_person_model is not None:
        tracker_options = {
            "detect_interval": 5,
            "scale_w": out_w,
            "use_face_regions": True,
            "max_missed": 10,
            "detect_on_first_frame": True,
        }
    multi_tracker = MultiFaceTracker(**tracker_options)
    resources.callback(multi_tracker.close)

    main_face_mesh = MainFaceMesh(max_num_faces=3, bbox_expand=0.05)
    resources.callback(main_face_mesh.close)

    if multi_person_model is None:
        pose_skeleton = PoseSkeleton()
    else:
        pose_skeleton = MultiPoseSkeleton(multi_person_model)
    resources.callback(pose_skeleton.close)

    frame_idx = 0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_sec = frame_idx / fps_video

        frame = cv2.resize(frame, (out_w, out_h))

        # ===== FX by time =====
        cfg = get_fx_config(current_sec)

        if cfg["fx"] == "original":
            fx = frame.copy()
            fx_name = "Original"

        elif cfg["fx"] == "edge":
            mode = cfg.get("mode")
            fx = edge_fx(frame, mode=mode)
            fx_name = f"Edge ({mode})"

        elif cfg["fx"] == "colormap":
            fx = color_fx(frame)
            fx_name = "ColorMap Grid (4x3)"

        elif cfg["fx"] == "geometry":

            mode = cfg.get("mode", "resize_rotate")

            if mode == "resize_rotate":
                duration = max(cfg["end"] - cfg["start"], 1e-6)
                progress = (current_sec - cfg["start"]) / duration
                progress = max(0.0, min(1.0, progress))

                fx = geometry_fx(frame, mode=mode, progress=progress)
                fx_name = f"Geometry ({mode}) {progress:.2f}"

            elif mode == "flip_4in1":
                fx = geometry_fx(frame, mode=mode)
                fx_name = f"Geometry ({mode})"


            else:
                fx = frame.copy()
                fx_name = f"Geometry (unknown: {mode})"

        elif cfg["fx"] == "feature":
            mode = cfg.get("mode")
            fx = feature_fx(frame, mode=mode, frame_idx=frame_idx)
            fx_name = f"Features ({mode})"

        # Multi-face tracking
        if 65 <= current_sec <= 78:
            face_regions = None
            if multi_person_model is not None:
                def face_regions():
                    return pose_skeleton.face_regions(frame, round(current_sec * 1000))
            tracks = multi_tracker.update(frame, face_regions)

            for t in tracks:
                x, y, w, h = t["bbox"]
                tid = t["id"]

                cv2.rectangle(
                    fx,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    fx,
                    f"ID {tid}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            fx_name = "Multi Face Tracking"

        # Main face mesh
        if 78 <= current_sec <= 115:
            fx, _ = main_face_mesh.apply(
                fx,
                current_sec=current_sec,
                hide_bbox_after=100.0,
                draw_mesh=True,
                enable_beauty=True,
            )
            fx_name = "MainFaceMesh & GlowFx"

        # Pose skeleton
        if 83 <= current_sec <= 132:
            if multi_person_model is None:
                fx = pose_skeleton.apply(fx)
                fx_name = "GlowFx & Pose Skeleton"
            else:
                fx = pose_skeleton.apply(fx, round(current_sec * 1000))
                fx_name = "GlowFx & Multi-Person Pose"

        # ===== FPS safe calc =====
        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        fps = 1.0 / dt
        prev_time = now

        # ===== UI =====
        draw_overlay_ui(fx, frame_idx, total, fx_name, fps)

        # ===== PiP =====
        if cfg["pip"]:
            scale = 0.25
            tw = int(out_w * scale)
            th = int(out_h * scale)

            thumb = cv2.resize(frame, (tw, th))

            fh, fw = fx.shape[:2]   # ⭐ 關鍵：以 fx 為準

            margin = 20
            x = margin
            y = fh - th - margin

            # 防呆（建議保留）
            if y < 0 or x < 0 or y + th > fh or x + tw > fw:
                print(f"[WARN] PiP out of bounds: fx=({fh},{fw}), pip=({th},{tw}), pos=({x},{y})")
            else:
                fx[y:y + th, x:x + tw] = thumb
                cv2.rectangle(
                    fx,
                    (x, y),
                    (x + tw, y + th),
                    (255, 255, 255),
                    2
                )
        # ===== Output =====
        writer.write(fx)
        if preview:
            cv2.imshow("Video FX", fx)
            if cv2.waitKey(1) == 27:
                raise VideoProcessingCancelled
    return frame_idx


def process_video(input_path, output_path, *, preview=True, multi_person_model=None):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.is_file():
        raise ValueError(f"Input video does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve() or (
        output_path.exists() and input_path.samefile(output_path)
    ):
        raise ValueError("Input and output must be different files.")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output already exists: {output_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Choose an output filename ending in .mp4.")
    if not output_path.parent.is_dir():
        raise ValueError(f"Output folder does not exist: {output_path.parent}")

    if multi_person_model is not None:
        multi_person_model = Path(multi_person_model).expanduser()
        if not multi_person_model.is_file():
            raise ValueError(f"Pose model does not exist: {multi_person_model}")

    cap = cv2.VideoCapture(str(input_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if any(not math.isfinite(value) or value <= 0 for value in (fps_video, width, height)):
            raise RuntimeError("Input video has invalid FPS or frame dimensions.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w = 960
        out_h = max(2, 2 * int(out_w * height / width / 2))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        with TemporaryDirectory(prefix=".opencv-render-", dir=output_path.parent) as temp_dir:
            temp_video = Path(temp_dir) / "video.mp4"
            writer = cv2.VideoWriter(str(temp_video), fourcc, fps_video, (out_w, out_h))
            try:
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open a video writer in: {output_path.parent}")
                with ExitStack() as resources:
                    frames_written = render_frames(
                        cap, writer, fps_video, total, out_w, out_h, preview,
                        resources, multi_person_model,
                    )
                if not frames_written:
                    raise RuntimeError("No frames were decoded from the input video.")
            finally:
                writer.release()

            result = cv2.VideoCapture(str(temp_video))
            try:
                readable, _ = result.read()
                if not readable:
                    raise RuntimeError("The generated video could not be decoded.")
            finally:
                result.release()

            # Publish atomically without replacing a file created during processing.
            try:
                os.link(temp_video, output_path)
            except FileExistsError as exc:
                raise FileExistsError(f"Output already exists: {output_path}") from exc
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Apply timed OpenCV effects to your video.")
    parser.add_argument("--input", required=True, help="Path to your video.")
    parser.add_argument("--output", required=True, help="Path to a new MP4 file.")
    parser.add_argument("--no-preview", action="store_true", help="Process without opening a window.")
    parser.add_argument("--multi-person", metavar="POSE_MODEL", help="Enable two-person pose with a MediaPipe .task model.")
    args = parser.parse_args()

    try:
        process_video(
            args.input, args.output, preview=not args.no_preview,
            multi_person_model=args.multi_person,
        )
    except VideoProcessingCancelled:
        print("Cancelled. No output saved.")
        return
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
