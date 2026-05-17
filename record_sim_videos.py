#!/usr/bin/env python3
"""
record_sim_videos.py
--------------------
Records two 20-second videos from a running pipe-inspection simulation.

  videos/cylinder_render.mp4  — 3D curved-pipe coverage, colour-mapped by
                                 inspection completion (blue → red).
  videos/crawler_camera.mp4   — Raw fisheye image from the crawler's camera
                                 (the Gazebo sensor, inside the pipe).

Run WHILE the simulation is already launched:

    cd ~/xolo
    source install/setup.bash
    python3 record_sim_videos.py                    # 20 s, default params
    python3 record_sim_videos.py --duration 30
    python3 record_sim_videos.py --output ~/Desktop
"""

import argparse
import math
import os
import threading
import time

import cv2
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


# ── pipe geometry defaults (overridden by --pipe_* flags) ─────────────────────
_DFLT_PIPE_RADIUS    = 0.15
_DFLT_PIPE_LENGTH    = 4.5
_DFLT_BEND_ANGLE_DEG = 90.0
_DFLT_DEPTH_PER_FRAME = 0.01   # updated live from odom_depth_node

# matplotlib surface resolution (higher = slower render but smoother)
_H_SURF = 60
_W_SURF = 45


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_pipe_surface(h, w, pipe_length, pipe_radius, bend_angle_deg, bend_radius):
    """Return (H, W, 3) float64 vertex grid for the full curved pipe."""
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


def _cov_to_rgba(cov: np.ndarray) -> np.ndarray:
    """
    Vectorised: (H, W) float32 [0,1] → (H, W, 4) RGBA  (same ramp as RViz node).
      0.0 – 0.1  →  dark blue
      0.1 – 0.5  →  blue → cyan
      0.5 – 1.0  →  cyan → red
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


def _fig_to_bgr(fig) -> np.ndarray:
    """Convert a matplotlib figure to an HxWx3 BGR uint8 array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    raw  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return raw.reshape(h, w, 3)[:, :, ::-1]   # RGB → BGR


def _add_hud(img: np.ndarray, label: str, elapsed: float, duration: float) -> np.ndarray:
    """Burn a small text HUD (label + progress bar) onto the frame."""
    out   = img.copy()
    h, w  = out.shape[:2]
    pct   = elapsed / duration
    bar_w = int(w * 0.6)
    bar_h = 10

    # progress bar background
    cv2.rectangle(out, (20, h - 40), (20 + bar_w, h - 40 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(out, (20, h - 40), (20 + int(bar_w * pct), h - 40 + bar_h),
                  (0, 200, 80), -1)
    cv2.putText(out, label, (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"{elapsed:4.1f} / {duration:.0f} s",
                (20 + bar_w + 10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return out


# ── ROS2 recorder node ────────────────────────────────────────────────────────

class RecorderNode(Node):
    def __init__(self, args):
        super().__init__("video_recorder_node")

        self._duration        = args.duration
        self._fps             = args.fps
        self._out_dir         = os.path.expanduser(args.output)
        self._pipe_radius     = args.pipe_radius
        self._pipe_length     = args.pipe_length
        self._bend_angle_deg  = args.bend_angle_deg
        self._depth_per_frame = _DFLT_DEPTH_PER_FRAME

        if self._bend_angle_deg > 0:
            self._bend_radius = (
                self._pipe_length / (self._bend_angle_deg * math.pi / 180.0)
            )
        else:
            self._bend_radius = 0.0

        os.makedirs(self._out_dir, exist_ok=True)

        self._bridge   = CvBridge()
        self._lock     = threading.Lock()
        self._cov_buf  = []    # [(elapsed_s, mono8_heatmap), ...]
        self._cam_buf  = []    # [(elapsed_s, bgr_image), ...]
        self._t0       = None
        self._done     = False

        self._sub_cov = self.create_subscription(
            Image, "/pipe/coverage",    self._cov_cb, 10)
        self._sub_cam = self.create_subscription(
            Image, "/camera/image_raw", self._cam_cb, 10)

        self.get_logger().info(
            f"VideoRecorder ready.\n"
            f"  Duration     : {self._duration} s\n"
            f"  Output dir   : {self._out_dir}\n"
            f"  Pipe shape   : length={self._pipe_length} m  "
            f"bend={self._bend_angle_deg}°  R_bend={self._bend_radius:.3f} m\n"
            f"Waiting for first /pipe/coverage frame …"
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cov_cb(self, msg: Image) -> None:
        t = time.monotonic()
        if self._t0 is None:
            self._t0 = t
            self.get_logger().info("Recording started.")
        elapsed = t - self._t0
        if elapsed > self._duration or self._done:
            return
        try:
            heatmap = self._bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception:
            return
        with self._lock:
            self._cov_buf.append((elapsed, heatmap.copy()))

    def _cam_cb(self, msg: Image) -> None:
        t = time.monotonic()
        if self._t0 is None or self._done:
            return
        elapsed = t - self._t0
        if elapsed > self._duration:
            return
        try:
            img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        with self._lock:
            self._cam_buf.append((elapsed, img.copy()))

    @property
    def recording_done(self) -> bool:
        return self._t0 is not None and (time.monotonic() - self._t0) >= self._duration

    # ── post-recording render ─────────────────────────────────────────────────

    def save_videos(self) -> None:
        self._done = True
        with self._lock:
            cov_buf = list(self._cov_buf)
            cam_buf = list(self._cam_buf)

        self.get_logger().info(
            f"Recording complete — "
            f"{len(cov_buf)} coverage frames, {len(cam_buf)} camera frames."
        )

        self._save_crawler_video(cam_buf)
        self._save_cylinder_video(cov_buf)

    # ── Video 2: crawler fisheye ──────────────────────────────────────────────

    def _save_crawler_video(self, buf) -> None:
        if not buf:
            self.get_logger().warn("No camera frames buffered — skipping crawler video.")
            return

        path = os.path.join(self._out_dir, "crawler_camera.mp4")
        h, w = buf[0][1].shape[:2]
        fps  = len(buf) / self._duration

        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for elapsed, frame in buf:
            frame_hud = _add_hud(frame, "Crawler camera (inside pipe)", elapsed, self._duration)
            out.write(frame_hud)
        out.release()

        self.get_logger().info(
            f"Saved: {path}  ({len(buf)} frames, {fps:.1f} fps)"
        )

    # ── Video 1: 3D cylinder coverage ────────────────────────────────────────

    def _save_cylinder_video(self, buf) -> None:
        if not buf:
            self.get_logger().warn("No coverage frames buffered — skipping cylinder video.")
            return

        path = os.path.join(self._out_dir, "cylinder_render.mp4")

        # Pre-build fixed pipe surface geometry  (H_SURF × W_SURF vertices)
        verts = _build_pipe_surface(
            _H_SURF, _W_SURF,
            self._pipe_length, self._pipe_radius,
            self._bend_angle_deg, self._bend_radius,
        )
        X = verts[:, :, 0]
        Y = verts[:, :, 1]
        Z = verts[:, :, 2]

        # Number of FACES = (H_SURF-1) × (W_SURF-1)  — matplotlib Poly3DCollection
        # set_facecolors needs exactly this many rows.
        n_faces = (_H_SURF - 1) * (_W_SURF - 1)

        # Maximum heatmap height across all frames — used to compute progress fraction
        max_rows = max(h.shape[0] for _, h in buf) if buf else 1

        # Nice viewing angle: from outside-above the bend
        dark = "#111111"
        fig  = plt.figure(figsize=(12, 9), dpi=100, facecolor=dark)
        ax   = fig.add_subplot(111, projection="3d", facecolor=dark)
        ax.set_axis_off()
        ax.view_init(elev=28, azim=-50)

        # Fixed axis limits (equal aspect)
        all_pts = verts.reshape(-1, 3)
        rng     = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0 * 1.2
        mid     = all_pts.mean(axis=0)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)

        # Title
        bend_txt = (
            f"{self._bend_angle_deg:.0f}° arc  R={self._bend_radius:.2f} m"
            if self._bend_radius > 0.01 else "straight"
        )
        fig.text(0.5, 0.96, f"Pipe inspection coverage — {bend_txt}",
                 ha="center", va="top", color="#cccccc",
                 fontsize=13, fontweight="bold")

        # Initial surface: all dark-blue (unseen)
        # plot_surface accepts facecolors of shape (H, W, 4) and internally maps
        # them to (H-1)×(W-1) faces.  We draw once here so surf exists.
        init_c = np.full((_H_SURF, _W_SURF, 4), [0.12, 0.12, 0.40, 1.0],
                         dtype=np.float32)
        surf   = ax.plot_surface(
            X, Y, Z,
            facecolors=init_c,
            shade=False,
            linewidth=0,
            antialiased=False,
        )
        fig.canvas.draw()
        wh = fig.canvas.get_width_height()   # (pixel_width, pixel_height)

        fps = len(buf) / self._duration
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, wh)

        n = len(buf)
        self.get_logger().info(
            f"Rendering {n} coverage frames → {path}  (may take ~1 min)"
        )

        for idx, (elapsed, heatmap) in enumerate(buf):
            h_rows = heatmap.shape[0]

            # ── Fraction of pipe the robot has reached ────────────────────────
            # Use the ratio of current rows to max rows seen in the full clip.
            # This gives a clean 0→1 progress independent of depth_per_frame drift.
            reached_frac = min(h_rows / max_rows, 1.0)

            # Row index in the surface grid up to which the robot has been
            reached_surf_row = max(1, int(reached_frac * (_H_SURF - 1)))

            # ── Build per-vertex colour grid (H_SURF × W_SURF) ───────────────
            # All rows start as dark-blue (unreached).
            vertex_colors = np.full(
                (_H_SURF, _W_SURF, 4), [0.12, 0.12, 0.40, 1.0], dtype=np.float32
            )

            if reached_surf_row >= 1 and h_rows >= 1:
                # Resize heatmap to cover only the REACHED surface rows.
                # cv2.resize(src, (dst_cols, dst_rows))
                hm_ds = cv2.resize(
                    heatmap,
                    (_W_SURF, reached_surf_row),
                    interpolation=cv2.INTER_AREA,
                )
                cov = hm_ds.astype(np.float32) / 255.0
                vertex_colors[:reached_surf_row] = _cov_to_rgba(cov)

            # ── Convert vertex colours → face colours ─────────────────────────
            # plot_surface has (H-1)×(W-1) quad faces.
            # Each face colour = average of its 4 corner vertex colours.
            fc = (
                vertex_colors[:-1, :-1]
                + vertex_colors[1:, :-1]
                + vertex_colors[:-1, 1:]
                + vertex_colors[1:, 1:]
            ) / 4.0                                      # shape (H-1, W-1, 4)

            # set_facecolors needs a flat (N, 4) array where N = n_faces
            surf.set_facecolors(fc.reshape(n_faces, 4))
            fig.canvas.draw()

            frame = _fig_to_bgr(fig)
            pct_txt = f"{reached_frac * 100:.0f}% scanned"
            frame = _add_hud(
                frame,
                f"3D pipe coverage  —  {pct_txt}",
                elapsed,
                self._duration,
            )
            out.write(frame)

            if (idx + 1) % 20 == 0 or (idx + 1) == n:
                self.get_logger().info(f"  {idx+1}/{n} frames")

        out.release()
        plt.close(fig)
        self.get_logger().info(f"Saved: {path}  ({n} frames, {fps:.1f} fps)")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record two 20-s videos from a running pipe-inspection sim."
    )
    parser.add_argument("--duration",       type=float, default=20.0,
                        help="Recording length in seconds  (default 20)")
    parser.add_argument("--fps",            type=float, default=10.0,
                        help="Output video frame rate       (default 10)")
    parser.add_argument("--output",         default="~/xolo/videos",
                        help="Output directory              (default ~/xolo/videos)")
    parser.add_argument("--pipe_radius",    type=float, default=_DFLT_PIPE_RADIUS)
    parser.add_argument("--pipe_length",    type=float, default=_DFLT_PIPE_LENGTH)
    parser.add_argument("--bend_angle_deg", type=float, default=_DFLT_BEND_ANGLE_DEG)
    args = parser.parse_args()

    rclpy.init()
    node = RecorderNode(args)

    try:
        while rclpy.ok() and not node.recording_done:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.save_videos()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"\nDone!  Videos saved to: {os.path.expanduser(args.output)}/")


if __name__ == "__main__":
    main()
