"""
coverage_mapper_node.py
-----------------------
ROS2 node that accumulates cylindrical-unwrap frames to track which
regions of the pipe interior have been inspected.

For each incoming unwrapped frame the node:
  1. Samples the outermost ring of the image (the pipe wall nearest to the
     camera) to extract a 1-D angular-coverage vector.
  2. Appends that vector as a new depth slice to the CoverageMapper.
  3. Publishes the growing heatmap and a scalar coverage percentage.

The heatmap image grows taller by one row per frame (height = number of
frames processed so far, width = 360).  The cylinder visualiser uses the
image height to compute the pipe length dynamically.

Subscriptions
-------------
    /pipe/unwrapped           (sensor_msgs/Image)   — from CylinderUnwrapNode

Publications
------------
    /pipe/coverage            (sensor_msgs/Image)   — binary uint8 heatmap
    /pipe/coverage_percentage (std_msgs/Float32)    — 0.0 – 100.0

Usage (after building the package):
    ros2 run pipe_perception coverage_mapper_node
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from pipe_perception.algorithms.coverage_mapper import CoverageMapper


class CoverageMapperNode(Node):
    """
    Accumulates /pipe/unwrapped frames and publishes a coverage heatmap
    plus a scalar coverage percentage.

    Parameters (ROS2)
    -----------------
    angle_bins : int
        Angular resolution of the internal grid.  Default: 360.
    outer_fraction : float
        Fraction of the unwrapped frame height used as the "near pipe wall"
        ring for angular coverage extraction.  0.3 means the outermost 30 %
        of rows.  Default: 0.30.
    """

    SUB_TOPIC     = "/pipe/unwrapped"
    PUB_COVERAGE  = "/pipe/coverage"
    PUB_PERCENT   = "/pipe/coverage_percentage"
    LOG_INTERVAL  = 30  # frames between coverage log lines

    def __init__(self) -> None:
        super().__init__("coverage_mapper_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("angle_bins",      360)
        self.declare_parameter("outer_fraction",  0.30)

        angle_bins: int      = int(self.get_parameter("angle_bins").value)
        outer_frac: float    = float(self.get_parameter("outer_fraction").value)
        self._outer_frac     = max(0.05, min(1.0, outer_frac))

        # ------------------------------------------------------------------ #
        # Algorithm + bridge
        # ------------------------------------------------------------------ #
        self._mapper      = CoverageMapper(angle_bins=angle_bins)
        self._bridge      = CvBridge()
        self._frame_count = 0

        # ------------------------------------------------------------------ #
        # Subscriber / Publishers
        # ------------------------------------------------------------------ #
        self._subscriber = self.create_subscription(
            Image,
            self.SUB_TOPIC,
            self._image_callback,
            qos_profile=10,
        )
        self._pub_heatmap = self.create_publisher(
            Image, self.PUB_COVERAGE, qos_profile=10
        )
        self._pub_percent = self.create_publisher(
            Float32, self.PUB_PERCENT, qos_profile=10
        )

        # ------------------------------------------------------------------ #
        # Startup log
        # ------------------------------------------------------------------ #
        self.get_logger().info(
            f"CoverageMapperNode ready.\n"
            f"  Subscribing   : {self.SUB_TOPIC}\n"
            f"  Publishing    : {self.PUB_COVERAGE}  |  {self.PUB_PERCENT}\n"
            f"  Angle bins    : {angle_bins}\n"
            f"  Outer ring    : {self._outer_frac * 100:.0f}% of frame height"
        )

    # ---------------------------------------------------------------------- #
    # Callback
    # ---------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        """Extract angular coverage from outer ring, accumulate, publish."""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge decode failed: {exc}")
            return

        # ------------------------------------------------------------------ #
        # Extract angular coverage from the outermost pipe-wall ring.
        # Row 0 of the unwrapped image = optical centre (looking forward).
        # Last row = r_max (pipe wall immediately beside the camera).
        # We use the outer `outer_fraction` of rows as the "current depth"
        # coverage snapshot.
        # ------------------------------------------------------------------ #
        h = frame.shape[0]
        outer_start   = max(0, int(h * (1.0 - self._outer_frac)))
        outer_slice   = frame[outer_start:, :]          # (H*frac, 360, 3)

        # Which angles (columns) have any non-black pixel in the outer ring?
        angular_coverage = np.any(outer_slice > 0, axis=(0, 2)).astype(np.uint8)

        try:
            self._mapper.add_depth_slice(angular_coverage)
        except ValueError as exc:
            self.get_logger().warning(f"add_depth_slice rejected: {exc}")
            return

        self._frame_count += 1

        # ------------------------------------------------------------------ #
        # Publish binary heatmap  (height grows by 1 each frame)
        # ------------------------------------------------------------------ #
        heatmap_u8 = self._build_heatmap()
        try:
            heatmap_msg = self._bridge.cv2_to_imgmsg(heatmap_u8, encoding="mono8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge encode failed: {exc}")
            return

        heatmap_msg.header = msg.header
        self._pub_heatmap.publish(heatmap_msg)

        # ------------------------------------------------------------------ #
        # Publish coverage percentage
        # ------------------------------------------------------------------ #
        coverage_pct = self._mapper.get_coverage_percentage()
        pct_msg      = Float32()
        pct_msg.data = float(coverage_pct)
        self._pub_percent.publish(pct_msg)

        # ------------------------------------------------------------------ #
        # Periodic log
        # ------------------------------------------------------------------ #
        if self._frame_count % self.LOG_INTERVAL == 0:
            self.get_logger().info(
                f"Frame {self._frame_count:6d} | "
                f"Depth slices: {self._mapper.depth_slices} | "
                f"Coverage: {coverage_pct:.1f}%"
            )

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _build_heatmap(self) -> np.ndarray:
        """
        Convert the hit-count map to a binary uint8 [0, 255] coverage mask.

        A cell that has been seen at least once maps to 255 (fully covered).
        A cell that has never been seen maps to 0 (gap).  This binary approach
        gives a crisp covered/uncovered display regardless of how many times
        each region was revisited.
        """
        hit_map = self._mapper.hit_map
        return np.where(hit_map > 0, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    node: CoverageMapperNode | None = None
    try:
        node = CoverageMapperNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
