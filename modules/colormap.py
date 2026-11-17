"""
Color effect using ColorMap
input:
    frame (BGR image to Gray, np.ndarray)
output:
    BGR image with Color effect
"""
import cv2
import numpy as np

def apply(frame, rows=4, cols=3):
    """
    Display multiple colormaps in a grid.
    Default: 4 x 3 = 12 colormaps
    """

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 12 種常用 colormap（順序可自己調）
    colormaps = [
        ("JET", cv2.COLORMAP_JET),
        ("HOT", cv2.COLORMAP_HOT),
        ("COOL", cv2.COLORMAP_COOL),
        ("HSV", cv2.COLORMAP_HSV),

        ("PLASMA", cv2.COLORMAP_PLASMA),
        ("MAGMA", cv2.COLORMAP_MAGMA),
        ("INFERNO", cv2.COLORMAP_INFERNO),
        ("TURBO", cv2.COLORMAP_TURBO),

        ("OCEAN", cv2.COLORMAP_OCEAN),
        ("PINK", cv2.COLORMAP_PINK),
        ("BONE", cv2.COLORMAP_BONE),
        ("RAINBOW", cv2.COLORMAP_RAINBOW),
    ]

    # define row & cloumn
    tile_h = h // rows
    tile_w = w // cols

    tiles = []
    idx = 0

    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            if idx < len(colormaps):
                name, cmap = colormaps[idx]
                color = cv2.applyColorMap(gray, cmap)
            else:
                name = ""
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            color = cv2.resize(color, (tile_w, tile_h))

            # ---------- label（左下角） ----------
            label_bg_h = 28
            cv2.rectangle(
                color,
                (0, tile_h - label_bg_h),
                (tile_w, tile_h),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                color,
                name,
                (8, tile_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            row_tiles.append(color)
            idx += 1

        tiles.append(np.hstack(row_tiles))

    grid = np.vstack(tiles)

    # 保險 resize 回原尺寸
    grid = cv2.resize(grid, (w, h))
    return grid
