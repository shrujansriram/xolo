"""
cylinder_unwrap.py
------------------
Converts flat camera frames into cylindrical unwrapped views via a
polar-to-rectangular coordinate transformation.

The angular sweep around the pipe axis becomes the horizontal axis
(columns 0–359 map to 0°–359°), and depth from the optical centre
becomes the vertical axis (row 0 = centre, last row = pipe wall).
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class CylinderUnwrapper:
    """
    Unwrap a fisheye / wide-angle pipe-inspection frame into a flat
    cylindrical projection.

    Parameters
    ----------
    radius : float
        Physical inner radius of the pipe in metres (default 0.15 m / 150 mm).
        Stored as metric metadata for downstream nodes; does not affect the
        pixel-level remapping.
    """

    def __init__(self, radius: float = 0.15) -> None:
        self.radius: float = radius

        # Cached remap look-up tables (rebuilt only when frame size changes)
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._cached_shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_remap_tables(self, h: int, w: int) -> None:
        """
        Pre-compute float32 remap LUTs for a frame of size (h, w).

        Layout of the output grid (H x 360):
          - Columns 0-359  -> angles 0-359 deg (counter-clockwise from 3 o'clock)
          - Rows 0 to H-1  -> radii from 0 (optical centre) to r_max (pipe wall)
        """
        cx: float = w * 0.5
        cy: float = h * 0.5

        # Largest inscribed-circle radius that stays inside valid pixels.
        # Subtract 1 px so bilinear sampling never reads outside the frame.
        r_max: float = min(cx, cy) - 1.0

        out_h, out_w = h, 360

        # angles: shape (360,)  — endpoint=False avoids duplicate 0/360 deg column
        angles = np.linspace(0.0, 2.0 * np.pi, out_w, endpoint=False)
        # radii:  shape (H,)    — row 0 lands on the optical centre
        radii = np.linspace(0.0, r_max, out_h)

        # Broadcast to (H, 360) grids
        angle_grid, radius_grid = np.meshgrid(angles, radii)

        # Cartesian source coordinates in the input frame
        self._map_x = (cx + radius_grid * np.cos(angle_grid)).astype(np.float32)
        self._map_y = (cy + radius_grid * np.sin(angle_grid)).astype(np.float32)
        self._cached_shape = (h, w)

    def _ensure_maps(self, h: int, w: int) -> None:
        """Rebuild LUTs only when the incoming frame size has changed."""
        if self._cached_shape != (h, w):
            self._build_remap_tables(h, w)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def unwrap_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Unwrap a flat BGR camera frame into a cylindrical projection.

        Parameters
        ----------
        frame : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.

        Returns
        -------
        np.ndarray
            Cylindrical projection, shape (H, 360, 3), dtype uint8.
            Pixels sampled outside the source frame are set to black (0, 0, 0).

        Raises
        ------
        ValueError
            If *frame* is None, empty, not 3-channel, or too small to unwrap.
        """
        # --- Input validation ---
        if frame is None or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image (H x W x 3), "
                f"got shape {frame.shape}."
            )

        h, w = frame.shape[:2]
        if h < 4 or w < 4:
            raise ValueError(
                f"Frame is too small for cylindrical unwrapping: {w}x{h} px. "
                f"Minimum size is 4x4."
            )

        # --- Build / reuse remap tables ---
        self._ensure_maps(h, w)

        # --- Remap: polar -> rectangular ---
        # INTER_LINEAR gives smooth results; BORDER_CONSTANT fills out-of-bounds
        # source coordinates with black so image-corner artefacts don't bleed in.
        unwrapped: np.ndarray = cv2.remap(
            frame,
            self._map_x,   # type: ignore[arg-type]
            self._map_y,   # type: ignore[arg-type]
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return unwrapped
