"""
coverage_mapper_node.py
-----------------------
ROS2 node that accumulates cylindrical-unwrap frames to track which
regions of the pipe interior have been inspected.

In odom-gated mode (use_odom=True, default for live simulation):
  A new depth slice is added only when the robot has physically moved
  at least `slice_distance_m` metres since the last slice.  The robot's
  actual pose at that moment is stored and published as a nav_msgs/Path
  on /pipe/waypoints so the cylinder_visualizer_3d_node can draw the
  3D cylinder along the robot's real trajectory.

In frame-gated mode (use_odom=False, for offline video playback):
  A new depth slice is added on every incoming /pipe/unwrapped frame,
  reproducing the previous behaviour used by video_inspect.launch.py.

Subscriptions
-------------
    /pipe/unwrapped           (sensor_msgs/Image)   — from CylinderUnwrapNode
    /odom                     (nav_msgs/Odometry)   — robot pose (odom-gated mode only)

Publications
------------
    /pipe/coverage            (sensor_msgs/Image)   — binary uint8 heatmap
    /pipe/coverage_percentage (std_msgs/Float32)    — 0.0 – 100.0
    /pipe/waypoints           (nav_msgs/Path)        — robot poses at each depth slice
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from pipe_perception.algorithms.coverage_mapper import CoverageMapper


# QoS for /pipe/waypoints — transient-local so late-joining subscribers
# (e.g. RViz, cylinder_visualizer) receive the full accumulated path.
_WAYPOINTS_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class CoverageMapperNode(Node):
    """
    Accumulates /pipe/unwrapped frames and publishes:
      /pipe/coverage            — binary heatmap (grows with each accepted slice)
      /pipe/coverage_percentage — scalar coverage %
      /pipe/waypoints           — robot poses at each accepted depth slice

    Parameters (ROS2)
    -----------------
    angle_bins : int
        Angular resolution.  Default: 360.
    outer_fraction : float
        Fraction of unwrapped-frame height used as the pipe-wall ring.
        Default: 0.30.
    use_odom : bool
        True  → odom-gated mode: one slice per slice_distance_m of travel.
        False → frame-gated mode: one slice per incoming frame.
        Default: True.
    slice_distance_m : float
        Minimum robot displacement (metres) between consecutive depth slices.
        Only used when use_odom=True.  Default: 0.02.
    """

    SUB_UNWRAPPED = "/pipe/unwrapped"
    SUB_ODOM      = "/odom"
    PUB_COVERAGE  = "/pipe/coverage"
    PUB_PERCENT   = "/pipe/coverage_percentage"
    PUB_WAYPOINTS = "/pipe/waypoints"
    LOG_INTERVAL  = 30

    def __init__(self) -> None:
        super().__init__("coverage_mapper_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("angle_bins",       360)
        self.declare_parameter("outer_fraction",   0.30)
        self.declare_parameter("use_odom",         True)
        self.declare_parameter("slice_distance_m", 0.02)

        angle_bins: int         = int(self.get_parameter("angle_bins").value)
        outer_frac: float       = float(self.get_parameter("outer_fraction").value)
        self._use_odom: bool    = bool(self.get_parameter("use_odom").value)
        self._slice_dist: float = float(self.get_parameter("slice_distance_m").value)
        self._outer_frac        = max(0.05, min(1.0, outer_frac))

        # ------------------------------------------------------------------ #
        # Algorithm + bridge
        # ------------------------------------------------------------------ #
        self._mapper      = CoverageMapper(angle_bins=angle_bins)
        self._bridge      = CvBridge()
        self._frame_count = 0   # frames received (not necessarily accepted)

        # ------------------------------------------------------------------ #
        # Odom-gating state
        # ------------------------------------------------------------------ #
        self._last_odom_x: float | None = None
        self._last_odom_y: float | None = None
        self._accum_dist:  float        = 0.0
        self._pending_pose: tuple | None = None   # (x, y, yaw) — set by odom_cb
        self._waypoints:   list          = []      # (x, y, yaw) at each slice

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._pub_heatmap  = self.create_publisher(Image,   self.PUB_COVERAGE,  10)
        self._pub_percent  = self.create_publisher(Float32, self.PUB_PERCENT,   10)
        self._pub_waypoints = self.create_publisher(
            Path, self.PUB_WAYPOINTS, _WAYPOINTS_QOS
        )

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self._sub_unwrapped = self.create_subscription(
            Image, self.SUB_UNWRAPPED, self._image_callback, 10
        )
        if self._use_odom:
            self._sub_odom = self.create_subscription(
                Odometry, self.SUB_ODOM, self._odom_callback, 10
            )

        # ------------------------------------------------------------------ #
        # Startup log
        # ------------------------------------------------------------------ #
        mode = (
            f"odom-gated  (slice_distance={self._slice_dist:.3f} m)"
            if self._use_odom
            else "frame-gated  (one slice per incoming frame)"
        )
        self.get_logger().info(
            f"CoverageMapperNode ready.\n"
            f"  Subscribing   : {self.SUB_UNWRAPPED}"
            + (f"  |  {self.SUB_ODOM}" if self._use_odom else "") + "\n"
            f"  Publishing    : {self.PUB_COVERAGE}  |  {self.PUB_PERCENT}"
            f"  |  {self.PUB_WAYPOINTS}\n"
            f"  Mode          : {mode}\n"
            f"  Angle bins    : {angle_bins}\n"
            f"  Outer ring    : {self._outer_frac * 100:.0f}% of frame height"
        )

    # ---------------------------------------------------------------------- #
    # Odom callback — maintain accumulated displacement + pending pose
    # ---------------------------------------------------------------------- #

    def _odom_callback(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        if self._last_odom_x is None:
            self._last_odom_x = x
            self._last_odom_y = y
            return

        step = math.hypot(x - self._last_odom_x, y - self._last_odom_y)
        self._accum_dist += step
        self._last_odom_x = x
        self._last_odom_y = y

        if self._accum_dist >= self._slice_dist:
            self._accum_dist   = 0.0
            self._pending_pose = (x, y, yaw)

    # ---------------------------------------------------------------------- #
    # Image callback — extract coverage, conditionally add slice
    # ---------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        """Decode unwrapped frame, extract outer-ring coverage, add slice if gating allows."""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge decode failed: {exc}")
            return

        self._frame_count += 1

        # ------------------------------------------------------------------ #
        # Odom gating: decide whether this frame produces a new depth slice
        # ------------------------------------------------------------------ #
        if self._use_odom:
            if self._pending_pose is None:
                return   # robot has not moved enough since the last slice
            pose = self._pending_pose
            self._pending_pose = None
        else:
            # Frame-gated mode: use a placeholder pose (0, 0, 0)
            pose = (0.0, 0.0, 0.0)

        # ------------------------------------------------------------------ #
        # Extract angular coverage from the outermost pipe-wall ring
        # ------------------------------------------------------------------ #
        h = frame.shape[0]
        outer_start   = max(0, int(h * (1.0 - self._outer_frac)))
        outer_slice   = frame[outer_start:, :]      # (H*frac, 360, 3)
        # Threshold > 30: rock pixels [22,14,8] (max=22) count as "not seen".
        # Pipe-wall pixels in the outer ring have lum ~110-200 (always > 30).
        angular_coverage = np.any(outer_slice > 30, axis=(0, 2)).astype(np.uint8)

        try:
            self._mapper.add_depth_slice(angular_coverage)
        except ValueError as exc:
            self.get_logger().warning(f"add_depth_slice rejected: {exc}")
            return

        # Store the pose for this depth slice
        self._waypoints.append(pose)

        # ------------------------------------------------------------------ #
        # Publish binary heatmap
        # ------------------------------------------------------------------ #
        heatmap_u8 = np.where(self._mapper.hit_map > 0, 255, 0).astype(np.uint8)
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
        # Publish waypoints path
        # ------------------------------------------------------------------ #
        self._publish_waypoints(msg.header.stamp)

        # ------------------------------------------------------------------ #
        # Periodic log
        # ------------------------------------------------------------------ #
        n_slices = self._mapper.depth_slices
        if n_slices % self.LOG_INTERVAL == 0:
            self.get_logger().info(
                f"Frame {self._frame_count:6d} | "
                f"Depth slices: {n_slices} | "
                f"Coverage: {coverage_pct:.1f}%"
            )

    # ---------------------------------------------------------------------- #
    # Publish accumulated waypoints as nav_msgs/Path
    # ---------------------------------------------------------------------- #

    def _publish_waypoints(self, stamp) -> None:
        path_msg            = Path()
        path_msg.header.stamp    = stamp
        path_msg.header.frame_id = "world"

        for x, y, yaw in self._waypoints:
            ps                       = PoseStamped()
            ps.header.stamp          = stamp
            ps.header.frame_id       = "world"
            ps.pose.position.x       = x
            ps.pose.position.y       = y
            ps.pose.position.z       = 0.0
            # Store yaw in quaternion (rotation about Z)
            ps.pose.orientation.x    = 0.0
            ps.pose.orientation.y    = 0.0
            ps.pose.orientation.z    = math.sin(yaw / 2.0)
            ps.pose.orientation.w    = math.cos(yaw / 2.0)
            path_msg.poses.append(ps)

        self._pub_waypoints.publish(path_msg)


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
