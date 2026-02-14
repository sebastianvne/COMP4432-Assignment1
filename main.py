import argparse
import os
import time
from typing import Callable, Tuple

import cv2
import numpy as np

PaletteArray = np.ndarray
DistanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

PALETTE_3_RGB: PaletteArray = np.array(
    [
        (0, 0, 0),          # black
        (128, 128, 128),    # gray
        (255, 255, 255),    # white
    ],
    dtype=np.uint8,
)

PALETTE_7_RGB: PaletteArray = np.array(
    [
        (0, 0, 0),          # black
        (255, 255, 255),    # white
        (255, 0, 0),        # red
        (255, 165, 0),      # orange
        (255, 255, 0),      # yellow
        (0, 255, 0),        # green
        (0, 0, 255),        # blue
    ],
    dtype=np.uint8,
)


def _compute_scaled_size(width: int, height: int, max_side: int) -> Tuple[int, int, float]:
    longest = max(width, height)
    if longest <= max_side:
        return width, height, 1.0
    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h, scale


def resize_keep_aspect(frame: np.ndarray, max_side: int = 100) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = frame.shape[:2]
    new_w, new_h, _ = _compute_scaled_size(width, height, max_side)
    if (new_w, new_h) == (width, height):
        return frame.copy(), (width, height)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (new_w, new_h)


def euclidean_distance_squared(pixels: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    diff = pixels[:, None, :] - palette_rgb[None, :, :]
    return np.sum(diff * diff, axis=2)


def map_to_palette(
    image_bgr: np.ndarray,
    palette_rgb: PaletteArray,
    distance_fn: DistanceFn = euclidean_distance_squared,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    palette_float = palette_rgb.astype(np.float32)

    dist_matrix = distance_fn(pixels, palette_float)
    nearest_idx = np.argmin(dist_matrix, axis=1).astype(np.int32)

    mapped_rgb = palette_rgb[nearest_idx].reshape(h, w, 3).astype(np.uint8)
    mapped_bgr = cv2.cvtColor(mapped_rgb, cv2.COLOR_RGB2BGR)
    return mapped_bgr, nearest_idx.reshape(h, w)


def build_brick_sizes(base_sizes: Tuple[Tuple[int, int], ...]) -> list[tuple[int, int]]:
    sizes: set[tuple[int, int]] = set()
    for w, h in base_sizes:
        sizes.add((w, h))
        sizes.add((h, w))
    return sorted(sizes, key=lambda s: (s[0] * s[1], max(s)), reverse=True)


def greedy_bricks(indices: np.ndarray, brick_sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int, int]]:
    rows, cols = indices.shape
    used = np.zeros((rows, cols), dtype=bool)
    bricks: list[tuple[int, int, int, int, int]] = []

    for y in range(rows):
        for x in range(cols):
            if used[y, x]:
                continue
            color = int(indices[y, x])
            placed = False
            for bw, bh in brick_sizes:
                if x + bw > cols or y + bh > rows:
                    continue
                if used[y : y + bh, x : x + bw].any():
                    continue
                if np.all(indices[y : y + bh, x : x + bw] == color):
                    used[y : y + bh, x : x + bw] = True
                    bricks.append((x, y, bw, bh, color))
                    placed = True
                    break
            if not placed:
                used[y, x] = True
                bricks.append((x, y, 1, 1, color))

    return bricks


def add_brick_outlines(
    image_bgr: np.ndarray,
    bricks: list[tuple[int, int, int, int, int]],
    grid_cols: int,
    grid_rows: int,
    outline_bgr: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    if thickness <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    scale_x = width / float(grid_cols)
    scale_y = height / float(grid_rows)

    outlined = image_bgr.copy()

    for x, y, bw, bh, _ in bricks:
        x0 = int(round(x * scale_x))
        x1 = int(round((x + bw) * scale_x))
        y0 = int(round(y * scale_y))
        y1 = int(round((y + bh) * scale_y))

        x0c = max(0, min(width, x0))
        x1c = max(0, min(width, x1))
        y0c = max(0, min(height, y0))
        y1c = max(0, min(height, y1))

        if x1c <= x0c or y1c <= y0c:
            continue

        outlined[y0c : min(y0c + thickness, y1c), x0c:x1c] = outline_bgr
        outlined[max(y1c - thickness, y0c) : y1c, x0c:x1c] = outline_bgr
        outlined[y0c:y1c, x0c : min(x0c + thickness, x1c)] = outline_bgr
        outlined[y0c:y1c, max(x1c - thickness, x0c) : x1c] = outline_bgr

    return outlined


def summarize_bricks(bricks: list[tuple[int, int, int, int, int]]) -> str:
    counts: dict[tuple[int, int], int] = {}
    for _, _, bw, bh, _ in bricks:
        key = (min(bw, bh), max(bw, bh))
        counts[key] = counts.get(key, 0) + 1

    total = len(bricks)
    ordered_keys = sorted(counts, key=lambda k: (k[0] * k[1], k[1], k[0]), reverse=True)

    lines = [f"Total bricks: {total}"]
    for w, h in ordered_keys:
        lines.append(f"{w}x{h}: {counts[(w, h)]}")
    return "\n".join(lines)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def capture_image(camera_index: int, output_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera (press s to save, q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            captured = frame.copy()
            cv2.imwrite(output_path, captured)
            break
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if captured is None:
        raise RuntimeError("No image captured.")
    return captured


def legoize_1x1(
    frame: np.ndarray,
    max_side: int,
    palette_rgb: PaletteArray,
    preview_scale: int,
    output_path: str,
) -> np.ndarray:
    resized, (resized_w, resized_h) = resize_keep_aspect(frame, max_side=max_side)
    mapped_bgr, _ = map_to_palette(resized, palette_rgb=palette_rgb)

    if preview_scale >= 2:
        mapped_bgr = cv2.resize(
            mapped_bgr,
            (resized_w * preview_scale, resized_h * preview_scale),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.imwrite(output_path, mapped_bgr)
    return mapped_bgr


def legoize_multisize(
    frame: np.ndarray,
    max_side: int,
    palette_rgb: PaletteArray,
    preview_scale: int,
    output_path: str,
    summary_path: str,
) -> Tuple[np.ndarray, str]:
    resized, (resized_w, resized_h) = resize_keep_aspect(frame, max_side=max_side)
    mapped_bgr, indices = map_to_palette(resized, palette_rgb=palette_rgb)

    brick_sizes = build_brick_sizes(((1, 1), (1, 2), (2, 2), (2, 4), (4, 2)))
    bricks = greedy_bricks(indices, brick_sizes)

    if preview_scale >= 2:
        rendered = cv2.resize(
            mapped_bgr,
            (resized_w * preview_scale, resized_h * preview_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        rendered = mapped_bgr

    outline_thickness = max(1, preview_scale // 6) if preview_scale >= 2 else 1
    rendered = add_brick_outlines(
        rendered,
        bricks=bricks,
        grid_cols=resized_w,
        grid_rows=resized_h,
        outline_bgr=(0, 0, 0),
        thickness=outline_thickness,
    )

    summary_text = summarize_bricks(bricks)
    cv2.imwrite(output_path, rendered)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    return rendered, summary_text


def live_lego(
    camera_index: int,
    max_side: int,
    palette_rgb: PaletteArray,
    preview_scale: int,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized, (resized_w, resized_h) = resize_keep_aspect(frame, max_side=max_side)
        mapped_bgr, indices = map_to_palette(resized, palette_rgb=palette_rgb)

        brick_sizes = build_brick_sizes(((1, 1), (1, 2), (2, 2), (2, 4), (4, 2)))
        bricks = greedy_bricks(indices, brick_sizes)

        if preview_scale >= 2:
            rendered = cv2.resize(
                mapped_bgr,
                (resized_w * preview_scale, resized_h * preview_scale),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            rendered = mapped_bgr

        outline_thickness = max(1, preview_scale // 6) if preview_scale >= 2 else 1
        rendered = add_brick_outlines(
            rendered,
            bricks=bricks,
            grid_cols=resized_w,
            grid_rows=resized_h,
            outline_bgr=(0, 0, 0),
            thickness=outline_thickness,
        )

        cv2.imshow("Original (press q to quit)", frame)
        cv2.imshow("LEGO Live (multi-size)", rendered)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW1 Vibe LEGO camera tasks")
    parser.add_argument(
        "--mode",
        choices=["capture", "task2", "task3", "live"],
        default="task2",
        help="capture: Task1, task2: 3-color 1x1, task3: multi-size, live: Task4",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--input", type=str, default="", help="Input image path")
    parser.add_argument("--max-side", type=int, default=100, help="Max side length (<=100)")
    parser.add_argument("--preview-scale", type=int, default=6, help="Scale up for display")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.max_side = 100
    args.preview_scale = 6
    args.output_dir = "outputs"
    args.camera = 0

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    ensure_dir(output_dir)

    timestamp = int(time.time())
    captured_path = os.path.join(output_dir, f"captured_{timestamp}.jpg")

    if args.mode == "capture":
        capture_image(args.camera, captured_path)
        return

    if args.mode == "live":
        live_lego(args.camera, args.max_side, PALETTE_7_RGB, args.preview_scale)
        return

    if args.input:
        frame = cv2.imread(args.input)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.input}")
    else:
        frame = capture_image(args.camera, captured_path)

    if args.mode == "task2":
        output_path = os.path.join(output_dir, f"lego_task2_{timestamp}.jpg")
        result = legoize_1x1(
            frame,
            max_side=min(args.max_side, 100),
            palette_rgb=PALETTE_3_RGB,
            preview_scale=args.preview_scale,
            output_path=output_path,
        )
        cv2.imshow("Task2 LEGO (3 colors)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    output_path = os.path.join(output_dir, f"lego_task3_{timestamp}.jpg")
    summary_path = os.path.join(output_dir, f"brick_summary_{timestamp}.txt")
    result, summary_text = legoize_multisize(
        frame,
        max_side=min(args.max_side, 100),
        palette_rgb=PALETTE_7_RGB,
        preview_scale=args.preview_scale,
        output_path=output_path,
        summary_path=summary_path,
    )
    print(summary_text)
    cv2.imshow("Task3 LEGO (multi-size)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
