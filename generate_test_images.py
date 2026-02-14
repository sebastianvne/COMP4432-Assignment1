"""
Generate test images of various sizes for robustness testing.
Usage: python generate_test_images.py [--output-dir OUTPUT_DIR]
"""
import argparse
import os

import cv2
import numpy as np


def generate_image(width: int, height: int, pattern: str = "gradient") -> np.ndarray:
    """Generate an image of given size with a test pattern."""
    if pattern == "gradient":
        # Horizontal gradient (R), vertical (G), diagonal (B)
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        r = np.tile(x, (height, 1))
        g = np.tile(y[:, None], (1, width))
        b = np.clip(np.arange(width) + np.arange(height)[:, None], 0, 255).astype(np.uint8)
        img = cv2.merge([r, g, b])
    elif pattern == "random":
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif pattern == "solid":
        img = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test images for robustness")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--pattern", choices=["gradient", "random", "solid"], default="gradient")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Small images
    small_sizes = [(10, 10), (20, 15), (50, 50), (1, 1), (5, 100), (100, 5)]
    for w, h in small_sizes:
        img = generate_image(w, h, args.pattern)
        path = os.path.join(output_dir, f"test_small_{w}x{h}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

    # Large images
    large_sizes = [(640, 480), (1920, 1080), (800, 600), (2000, 1500)]
    for w, h in large_sizes:
        img = generate_image(w, h, args.pattern)
        path = os.path.join(output_dir, f"test_large_{w}x{h}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

    # Boundary: exactly 100x100
    img = generate_image(100, 100, args.pattern)
    path = os.path.join(output_dir, "test_boundary_100x100.jpg")
    cv2.imwrite(path, img)
    print(f"Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    main()
