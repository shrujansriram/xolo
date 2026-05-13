"""
coverage_mapper.py
------------------
Tracks which regions of the pipe interior have been inspected.

The pipe surface is modelled as a 2-D grid:
  - Rows    (depth_slices): one entry per camera frame, representing the
             axial position along the pipe.  Row 0 = pipe entry, row N =
             deepest point reached so far.  The grid grows dynamically as
             new frames arrive.
  - Columns (angle_bins):   angular slices around the circumference (0–359°).

Each cell is binary: 1 = that angle was seen at that depth, 0 = gap.
Coverage percentage is the fraction of cells set to 1.
"""

from __future__ import annotations

from typing import List

import numpy as np


class CoverageMapper:
    """
    Accumulate per-frame angular coverage vectors and report inspection coverage.

    The grid height (number of depth slices) grows with every call to
    add_depth_slice().  Callers do not need to know the pipe length in advance.

    Parameters
    ----------
    angle_bins : int
        Number of angular bins (columns).  Matches the output width of
        CylinderUnwrapper, which always produces 360-column images.
        Default: 360.
    """

    def __init__(self, angle_bins: int = 360) -> None:
        if angle_bins < 1:
            raise ValueError(f"angle_bins must be a positive integer, got {angle_bins}.")

        self.angle_bins: int = angle_bins
        self._rows: List[np.ndarray] = []   # grows with each frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_depth_slice(self, angular_coverage: np.ndarray) -> None:
        """
        Append one depth slice of angular coverage.

        Each call represents the camera advancing one frame further into the
        pipe.  The row records which angles around the circumference were
        visible at that depth.

        Parameters
        ----------
        angular_coverage : np.ndarray
            1-D array of length angle_bins (or any length — it is
            nearest-neighbour resampled if the size differs).
            Non-zero values indicate that angle was seen.

        Raises
        ------
        ValueError
            If *angular_coverage* is None or empty.
        """
        if angular_coverage is None or angular_coverage.size == 0:
            raise ValueError("angular_coverage must be a non-empty 1-D array.")

        row = (angular_coverage > 0).astype(np.uint8)

        if row.size != self.angle_bins:
            # Nearest-neighbour resample to angle_bins width
            idx = (
                np.arange(self.angle_bins, dtype=np.float32)
                * (row.size / self.angle_bins)
            ).astype(np.int32)
            idx = np.clip(idx, 0, row.size - 1)
            row = row[idx]

        self._rows.append(row)

    def get_coverage_percentage(self) -> float:
        """
        Return the percentage of the pipe surface seen at least once.

        Returns
        -------
        float
            Coverage in the range [0.0, 100.0].
        """
        if not self._rows:
            return 0.0
        total = len(self._rows) * self.angle_bins
        seen  = int(sum(np.count_nonzero(r) for r in self._rows))
        return float(seen) / float(total) * 100.0

    def get_gaps(self, threshold: int = 0) -> np.ndarray:
        """
        Return a boolean mask of uninspected cells.

        Parameters
        ----------
        threshold : int
            Unused (kept for API compatibility).  A cell is a gap if it
            has never been seen.

        Returns
        -------
        np.ndarray
            Boolean array of shape (depth_slices, angle_bins).
            True  → gap  (never seen)
            False → covered
        """
        return self.hit_map == 0

    def reset(self) -> None:
        """Clear all accumulated depth slices."""
        self._rows.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hit_map(self) -> np.ndarray:
        """
        Read-only (depth_slices × angle_bins) uint8 array.

        1 = angle seen at that depth, 0 = not seen.
        Returns a (1 × angle_bins) zero array if no slices have been added.
        """
        if not self._rows:
            arr = np.zeros((1, self.angle_bins), dtype=np.uint8)
        else:
            arr = np.array(self._rows, dtype=np.uint8)
        arr.flags.writeable = False
        return arr

    @property
    def depth_slices(self) -> int:
        """Number of camera depth positions accumulated so far."""
        return len(self._rows)
