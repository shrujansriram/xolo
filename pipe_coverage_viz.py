#!/usr/bin/env python3
"""
pipe_coverage_viz.py
--------------------
Offline pipeline: reads a pipe inspection video and renders a 3D coverage
animation showing which regions of the pipe interior have been seen.

Steps:
  1. Read every frame of the input video
  2. Unwrap each fisheye frame into a cylindrical projection (polar → rect)
  3. Sample the outermost ring of each unwrapped frame → 360-bin angular coverage
  4. Accumulate into a growing heatmap (rows = frames, cols = angles 0–359°)
  5. Render a 3D matplotlib animation of the coverage growing over the pipe surface

Output: <output_dir>/pipe_coverage_render.mp4

No ROS2 required — runs standalone with cv2, numpy, and matplotlib.

Usage:
  python3 pipe_coverage_viz.py
  python3 pipe_coverage_viz.py --input videos/pipe.mp4
  python3 pipe_coverage_viz.py --input videos/pipe.mp4 --bend_angle_deg 0
  python3 pipe_coverage_viz.py --pipe_length 3.0 --bend_angle_deg 45
  python3 pipe_coverage_viz.py --help
"""

import argparse
import math
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np


# ── surface mesh resolution ────────────────────────────────────────────────────
_H_SURF = 60   # longitudinal grid rows  (higher = smoother, slower)
_W_SURF = 45   # circumferential columns (higher = smoother, slower)


# ── geometry: pipe surface vertex grid ────────────────────────────────────────

def _build_pipe_surface(
    h: int, w: int,
    pipe_length: float, pipe_radius: float,
    bend_angle_deg: float, bend_radius: float,
) -> np.ndarray:
    """
    Return an (H, W, 3) float64 array of world-space (x, y, z) vertices
    covering the full pipe surface.

    Curved pipe uses the Frenet-frame arc geometry (same as the RViz node):
      C(φ) = (R·sin(φ),  R·(1−cos(φ)),  0)
      e_v  = (0, 0, 1)   e_h = (sin(φ), −cos(φ), 0)
      P(φ,θ) = C + r·(cos(θ)·e_v + sin(θ)·e_h)

    Straight pipe runs along the world +X axis.
    """
    if bend_radius > 0.01:
        R       = bend_radius
        phi_max = math.radians(bend_angle_deg)
        s       = np.linspace(0.0, pipe_length, h)
        s_arc   = np.minimum(s, R * phi_max)
        s_ext   = np.maximum(s - R * phi_max, 0.0)
        phi     = s_arc / R
        cx      = R * np.sin(phi) + s_ext * np.cos(phi)
        cy      = R * (1.0 - np.cos(phi)) + s_ext * np.sin(phi)
        theta   = np.linspace(0.0, 2.0 * math.pi, w, endpoint=False)
        sp      = np.sin(phi)[:, None]
        cp      = np.cos(phi)[:, None]
        st      = np.sin(theta)[None, :]
        ct      = np.cos(theta)[None, :]
        r       = pipe_radius
        xs = cx[:, None] + r * st * sp
        ys = cy[:, None] + r * st * (-cp)
        zs = r * ct * np.ones((h, 1))
    else:
        angles = np.linspace(0.0, 2.0 * math.pi, w, endpoint=False)
        xs = np.tile(np.linspace(0.0, pipe_length, h)[:, None], (1, w))
        ys = pipe_radius * np.cos(np.tile(angles, (h, 1)))
        zs = pipe_radius * np.sin(np.tile(angles, (h, 1)))

    return np.stack([xs, ys, zs], axis=-1)


# ── colour mapping ─────────────────────────────────────────────────────────────

def _cov_to_rgba(cov: np.ndarray) -> np.ndarray:
    """
    Vectorised: (H, W) float32 [0, 1] → (H, W, 4) RGBA.

    Colour ramp (matches the RViz cylinder_visualizer_3d_node):
      0.0 – 0.1  →  dark blue  (unseen)
      0.1 – 0.5  →  blue → cyan
      0.5 – 1.0  →  cyan → red (fully inspected)
    """
    rgba = np.zeros((*cov.shape, 4), dtype=np.float32)
    rgba[..., 3] = 1.0

    m1 = cov <= 0.1
    rgba[m1] = [0.2, 0.2, 0.8, 1.0]

    m2 = (cov > 0.1) & (cov <= 0.5)
    t2 = (cov[m2] - 0.1) / 0.4
    rgba[m2, 0] = 0.2
    rgba[m2, 1] = 0.2 + t2 * 0.6
    rgba[m2, 2] = 0.8
    rgba[m2, 3] = 1.0

    m3 = cov > 0.5
    t3 = (cov[m3] - 0.5) / 0.5
    rgba[m3, 0] = 0.8 + t3 * 0.2
    rgba[m3, 1] = 0.2 - t3 * 0.2
    rgba[m3, 2] = 0.2
    rgba[m3, 3] = 1.0

    return rgba


# ── matplotlib / video helpers ─────────────────────────────────────────────────

def _fig_to_bgr(fig) -> np.ndarray:
    """Convert a matplotlib figure to an HxWx3 BGR uint8 array."""
    fig.canvas.draw()
    pw, ph = fig.canvas.get_width_height()
    raw = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return raw.reshape(ph, pw, 3)[:, :, ::-1]   # RGB → BGR


def _add_hud(img: np.ndarray, label: str, elapsed: float, duration: float) -> np.ndarray:
    """Burn a text HUD (label + progress bar) onto the frame."""
    out   = img.copy()
    h, w  = out.shape[:2]
    pct   = min(elapsed / max(duration, 1e-6), 1.0)
    bar_w = int(w * 0.6)
    bar_h = 10
    cv2.rectangle(out, (20, h - 40), (20 + bar_w, h - 40 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(out, (20, h - 40), (20 + int(bar_w * pct), h - 40 + bar_h),
                  (0, 200, 80), -1)
    cv2.putText(out, label, (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"{elapsed:4.1f} / {duration:.0f} s",
                (20 + bar_w + 10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return out


# ── fisheye unwrap (no ROS2 dependency) ───────────────────────────────────────

def _build_remap_tables(h: int, w: int):
    """
    Pre-compute polar→rectangular remap LUTs for a fisheye frame of size (h, w).

    Output: (h, 360) unwrapped image where:
      - columns 0–359 → angles 0–359° around the pipe axis
      - rows 0..h-1   → radius from optical centre to pipe wall
    """
    cx, cy = w * 0.5, h * 0.5
    r_max  = min(cx, cy) - 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    radii  = np.linspace(0.0, r_max, h)
    angle_grid, radius_grid = np.meshgrid(angles, radii)
    map_x = (cx + radius_grid * np.cos(angle_grid)).astype(np.float32)
    map_y = (cy + radius_grid * np.sin(angle_grid)).astype(np.float32)
    return map_x, map_y


def _unwrap_frame(frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Apply pre-computed LUTs to unwrap one fisheye frame."""
    return cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# ── phase 1: process video ─────────────────────────────────────────────────────

def process_video(video_path: str, outer_fraction: float) -> tuple:
    """
    Read all frames from *video_path*, unwrap each frame, extract angular
    coverage from the outermost ring, and accumulate a growing heatmap.

    Returns
    -------
    snapshots : list of (elapsed_s: float, heatmap: np.ndarray uint8)
        Heatmap snapshots taken every ~200th of the total frames.
    video_fps : float
    total_frames : int
        Number of frames successfully read from the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input video : {total_frames} frames @ {video_fps:.1f} fps  "
          f"({total_frames / video_fps:.1f} s)")

    # How often to save a snapshot for the output animation (~200 frames max)
    snapshot_every = max(1, total_frames // 200)

    map_x = map_y = None   # built lazily once we know the frame size
    rows: list     = []    # growing list of 360-bin coverage row vectors
    snapshots: list = []
    frames_read    = 0



    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1

        # Build remap LUTs once (reused for every frame of the same size)
        if map_x is None:
            map_x, map_y = _build_remap_tables(frame.shape[0], frame.shape[1])

        unwrapped = _unwrap_frame(frame, map_x, map_y)

        # Sample the outer ring (the pipe wall at the camera's current position)
        h = unwrapped.shape[0]
        outer_start = max(0, int(h * (1.0 - outer_fraction)))
        outer_slice = unwrapped[outer_start:]               # (H*frac, 360, 3)

        # An angle is "seen" if any pixel in the outer ring is non-black
        angular_cov = np.any(outer_slice > 0, axis=(0, 2)).astype(np.uint8)
        rows.append(angular_cov)

        # Save a snapshot of the heatmap state
        if idx % snapshot_every == 0 or idx == total_frames - 1:
            hit_map = np.array(rows, dtype=np.uint8)
            heatmap = np.where(hit_map > 0, 255, 0).astype(np.uint8)
            elapsed = idx / video_fps
            snapshots.append((elapsed, heatmap.copy()))

        if (idx + 1) % 200 == 0 or idx == total_frames - 1:
            print(f"  Processed {idx + 1:>5}/{total_frames} frames  "
                  f"| snapshots: {len(snapshots)}")

    cap.release()
    print(f"Processing complete: {frames_read} frames read, "
          f"{len(snapshots)} snapshots saved.")
    return snapshots, video_fps, frames_read


# ── phase 2: render 3D animation ──────────────────────────────────────────────

def render_coverage_video(
    snapshots: list,
    output_path: str,
    pipe_radius: float,
    pipe_length: float,
    bend_angle_deg: float,
    bend_radius: float,
    video_fps: float,
    total_frames: int,
    output_fps: float = 30.0,
) -> None:
    """
    Render an animated 3D view of coverage growing over the pipe surface.

    Each snapshot becomes one output frame.  The pipe surface starts dark-blue
    (unseen) and transitions through cyan to red as coverage accumulates.
    """
    if not snapshots:
        print("No snapshots — nothing to render.", file=sys.stderr)
        return

    # Build fixed pipe geometry (done once — the shape doesn't change)
    verts = _build_pipe_surface(
        _H_SURF, _W_SURF, pipe_length, pipe_radius, bend_angle_deg, bend_radius,
    )
    X, Y, Z = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]
    n_faces = (_H_SURF - 1) * (_W_SURF - 1)

    video_duration = total_frames / max(video_fps, 1e-6)
    max_rows       = max(h.shape[0] for _, h in snapshots)

    # ── set up figure ──────────────────────────────────────────────────────────
    dark = "#111111"
    fig  = plt.figure(figsize=(12, 9), dpi=100, facecolor=dark)
    ax   = fig.add_subplot(111, projection="3d", facecolor=dark)
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-50)

    # Equal-aspect axis limits
    all_pts = verts.reshape(-1, 3)
    rng     = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0 * 1.2
    mid     = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)

    bend_txt = (
        f"{bend_angle_deg:.0f}° arc  R={bend_radius:.2f} m"
        if bend_radius > 0.01 else "straight"
    )
    fig.text(
        0.5, 0.96, f"Pipe inspection coverage — {bend_txt}",
        ha="center", va="top", color="#cccccc", fontsize=13, fontweight="bold",
    )

    # Draw the surface once in all-unseen colour so the surf object exists
    init_c = np.full((_H_SURF, _W_SURF, 4), [0.12, 0.12, 0.40, 1.0], dtype=np.float32)
    surf   = ax.plot_surface(X, Y, Z, facecolors=init_c,
                             shade=False, linewidth=0, antialiased=False)
    fig.canvas.draw()
    wh = fig.canvas.get_width_height()   # (pixel_width, pixel_height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, wh)
    if not out.isOpened():
        print(f"ERROR: could not open VideoWriter for {output_path}", file=sys.stderr)
        plt.close(fig)
        return

    n = len(snapshots)
    print(f"Rendering {n} frames → {output_path}")

    for idx, (elapsed, heatmap) in enumerate(snapshots):
        h_rows = heatmap.shape[0]

        # How far along the pipe has the camera reached?
        reached_frac     = min(h_rows / max_rows, 1.0)
        reached_surf_row = max(1, int(reached_frac * (_H_SURF - 1)))

        # Build per-vertex colour grid: start all dark-blue (unreached)
        vertex_colors = np.full(
            (_H_SURF, _W_SURF, 4), [0.12, 0.12, 0.40, 1.0], dtype=np.float32
        )
        if reached_surf_row >= 1 and h_rows >= 1:
            # Resize heatmap only to the reached surface rows (not the full surface)
            hm_ds = cv2.resize(
                heatmap, (_W_SURF, reached_surf_row), interpolation=cv2.INTER_AREA
            )
            cov = hm_ds.astype(np.float32) / 255.0
            vertex_colors[:reached_surf_row] = _cov_to_rgba(cov)

        # Average 4 corner vertex colours → per-face colour (H-1)×(W-1)
        fc = (
            vertex_colors[:-1, :-1]
            + vertex_colors[1:, :-1]
            + vertex_colors[:-1, 1:]
            + vertex_colors[1:, 1:]
        ) / 4.0

        surf.set_facecolors(fc.reshape(n_faces, 4))   # exactly n_faces rows
        fig.canvas.draw()

        frame_img = _fig_to_bgr(fig)
        cov_pct   = float((heatmap > 0).sum()) / float(max(heatmap.size, 1)) * 100.0
        frame_img = _add_hud(
            frame_img,
            f"Coverage: {cov_pct:.1f}%  |  depth slices: {h_rows}",
            elapsed,
            video_duration,
        )
        out.write(frame_img)

        if (idx + 1) % 20 == 0 or (idx + 1) == n:
            print(f"  {idx + 1:>4}/{n} frames rendered")

    out.release()
    plt.close(fig)
    print(f"Saved: {output_path}  ({n} frames @ {output_fps:.1f} fps)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input  = os.path.join(script_dir, "videos", "pipe.mp4")
    default_output = os.path.join(script_dir, "videos")

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", default=default_input,
        help="Path to the input pipe inspection video (default: videos/pipe.mp4)",
    )
    p.add_argument(
        "--output", default=default_output,
        help="Directory to write pipe_coverage_render.mp4 (default: videos/)",
    )
    p.add_argument(
        "--pipe_radius", type=float, default=0.15,
        help="Inner pipe radius in metres (default: 0.15)",
    )
    p.add_argument(
        "--pipe_length", type=float, default=4.5,
        help="Pipe arc length in metres (default: 4.5)",
    )
    p.add_argument(
        "--bend_angle_deg", type=float, default=90.0,
        help="Pipe bend in degrees: 0=straight, 90=L-bend, 180=U-bend (default: 90)",
    )
    p.add_argument(
        "--outer_fraction", type=float, default=0.5,
        help="Outer ring fraction of unwrapped frame used for coverage (default: 0.5)",
    )
    p.add_argument(
        "--output_fps", type=float, default=30.0,
        help="Output video frame rate (default: 30)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input video not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "pipe_coverage_render.mp4")

    bend_radius = (
        args.pipe_length / (args.bend_angle_deg * math.pi / 180.0)
        if args.bend_angle_deg > 0.0 else 0.0
    )

    print("\n=== Pipe Coverage Visualizer ===")
    print(f"  Input        : {args.input}")
    print(f"  Output       : {out_path}")
    print(f"  Pipe radius  : {args.pipe_radius} m")
    print(f"  Pipe length  : {args.pipe_length} m")
    print(f"  Bend angle   : {args.bend_angle_deg}°  (R_bend={bend_radius:.3f} m)")
    print(f"  Outer frac   : {args.outer_fraction:.0%}  "
          f"(outermost portion of unwrapped frame)")
    print(f"  Output FPS   : {args.output_fps}")
    print()

    print("--- Phase 1: Processing video frames ---")
    snapshots, video_fps, total_frames = process_video(args.input, args.outer_fraction)

    print("\n--- Phase 2: Rendering 3D coverage animation ---")
    render_coverage_video(
        snapshots      = snapshots,
        output_path    = out_path,
        pipe_radius    = args.pipe_radius,
        pipe_length    = args.pipe_length,
        bend_angle_deg = args.bend_angle_deg,
        bend_radius    = bend_radius,
        video_fps      = video_fps,
        total_frames   = total_frames,
        output_fps     = args.output_fps,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
