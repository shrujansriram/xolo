"""
generate_test_video.py
----------------------
Generates a synthetic pipe inspection video.

The simulated camera looks forward down a pipe.  A 90-degree spotlight
sweeps from 0° to 270° around the circumference during the video, so at
the end exactly 75% of the ring is illuminated (red on the cylinder) and
25% remains permanently unseen (blue gap).

The spotlight centre goes from 45° to 225°, so the left edge starts at 0°
and the right edge ends at 270°.  No wrap around 0°/360°, so coverage
counts are uniform across all seen angles.

Output: ~/inspectly/videos/synthetic_pipe.mp4
"""

import os
import cv2
import numpy as np

OUTPUT_PATH = os.path.expanduser("~/inspectly/videos/synthetic_pipe.mp4")
NUM_FRAMES  = 300    # 10 seconds at 30 fps
SIZE        = 480    # square frame (480 x 480)
FPS         = 30.0

SPOTLIGHT_WIDTH_DEG = 90    # arc width of the simulated torch
# Centre sweeps 45° → 225°: left edge 0°, right edge 270°. No wrapping.
CENTER_START_DEG    = 45.0
CENTER_END_DEG      = 225.0


def generate(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try H.264 first, fall back to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, FPS, (SIZE, SIZE))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, FPS, (SIZE, SIZE))

    cx, cy = SIZE // 2, SIZE // 2
    r_max  = cx - 1          # 239 px — largest inscribed-circle radius

    # Pre-compute per-pixel polar coordinates (vectorised, done once)
    yy, xx      = np.mgrid[0:SIZE, 0:SIZE]
    r           = np.hypot(xx - cx, yy - cy).astype(np.float32)
    theta_norm  = (np.arctan2(yy - cy, xx - cx) + np.pi) % (2 * np.pi)
    in_circle   = r <= r_max

    sw_rad = np.deg2rad(SPOTLIGHT_WIDTH_DEG)

    rng = np.random.default_rng(seed=42)   # reproducible noise

    for idx in range(NUM_FRAMES):
        t = idx / max(NUM_FRAMES - 1, 1)   # 0.0 → 1.0

        # Spotlight centre moves from CENTER_START to CENTER_END — no wrapping
        center_deg = CENTER_START_DEG + t * (CENTER_END_DEG - CENTER_START_DEG)
        center_rad = np.deg2rad(center_deg)
        start = center_rad - sw_rad / 2   # always ≥ 0 (left edge = 0° at t=0)
        end   = center_rad + sw_rad / 2   # always ≤ 4.71 rad (right edge = 270° at t=1)

        # No wrap needed — start and end are both within [0, 2π]
        lit = in_circle & (theta_norm >= start) & (theta_norm <= end)

        # Base pipe-wall colour (grey-brown concrete)
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        frame[lit] = (58, 52, 47)          # BGR

        # Subtle per-pixel noise so the surface isn't flat
        noise      = rng.integers(0, 25, (SIZE, SIZE, 3), dtype=np.uint8)
        frame[lit] = np.clip(
            frame[lit].astype(np.int16) + noise[lit].astype(np.int16),
            5, 255
        ).astype(np.uint8)

        out.write(frame)

    out.release()
    print(f"Done — {NUM_FRAMES} frames written to {output_path}")


if __name__ == "__main__":
    generate(OUTPUT_PATH)
