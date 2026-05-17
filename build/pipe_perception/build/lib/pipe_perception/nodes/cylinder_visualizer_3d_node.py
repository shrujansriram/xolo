"""
cylinder_visualizer_3d_node.py
-------------------------------
ROS2 node that converts the growing 2D coverage heatmap into a 3D
cylindrical TRIANGLE_LIST Marker for display in RViz2.

The cylinder shape follows the robot's actual trajectory from odometry.
Cross-sections are placed at the (x, y, yaw) poses stored in
/pipe/waypoints, so the 3D map accurately reflects where the robot
physically travelled — including straight runs, corners, and bends.

Subscriptions
-------------
    /pipe/coverage   (sensor_msgs/Image)  — mono8 heatmap from CoverageMapperNode
    /pipe/waypoints  (nav_msgs/Path)      — robot poses at each depth slice

Publications
------------
    /visualization/cylinder  (visualization_msgs/Marker) — TRIANGLE_LIST

Usage (after building the package):
    ros2 run pipe_perception cylinder_visualizer_3d_node
    ros2 run pipe_perception cylinder_visualizer_3d_node --ros-args \
        -p pipe_radius:=0.15
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


# QoS matching the publisher in CoverageMapperNode — transient-local so
# a late-joining visualizer receives the full accumulated path on connect.
_WAYPOINTS_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class CylinderVisualizer3DNode(Node):
    """
    Rebuilds the 3D cylinder marker on every coverage heatmap update.

    The cylinder shape is driven by /pipe/waypoints (actual robot poses).
    No arc geometry is hardcoded — the marker faithfully follows the path
    the robot travelled, freezing cleanly when the robot stops.

    Parameters (ROS2)
    -----------------
    pipe_radius : float
        Cylinder radius in metres.  Default: 0.15
    grid_step : int
        Downsample factor applied to the heatmap before building the mesh.
        grid_step=1 → full resolution (very slow); grid_step=4 → 16× fewer
        triangles, good balance of speed and detail.  Default: 4
    """

    SUB_COVERAGE  = "/pipe/coverage"
    SUB_WAYPOINTS = "/pipe/waypoints"
    PUB_TOPIC     = "/visualization/cylinder"
    FRAME_ID      = "world"
    LOG_INTERVAL  = 10

    _ALPHA = 1.0   # fully opaque

    def __init__(self) -> None:
        super().__init__("cylinder_visualizer_3d_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("pipe_radius", 0.15)
        self.declare_parameter("grid_step",   4)

        self._pipe_radius: float = float(self.get_parameter("pipe_radius").value)
        self._grid_step: int     = max(1, int(self.get_parameter("grid_step").value))

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._bridge      = CvBridge()
        self._frame_count = 0
        self._waypoints: list = []   # list of (x, y, yaw) — one per depth slice

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self._sub_coverage = self.create_subscription(
            Image,
            self.SUB_COVERAGE,
            self._image_callback,
            qos_profile=10,
        )
        self._sub_waypoints = self.create_subscription(
            Path,
            self.SUB_WAYPOINTS,
            self._path_callback,
            qos_profile=_WAYPOINTS_QOS,
        )

        # ------------------------------------------------------------------ #
        # Publisher
        # ------------------------------------------------------------------ #
        self._publisher = self.create_publisher(
            Marker, self.PUB_TOPIC, qos_profile=10
        )

        # ------------------------------------------------------------------ #
        # Startup log
        # ------------------------------------------------------------------ #
        self.get_logger().info(
            f"CylinderVisualizer3DNode ready.\n"
            f"  Subscribing    : {self.SUB_COVERAGE}  |  {self.SUB_WAYPOINTS}\n"
            f"  Publishing     : {self.PUB_TOPIC}\n"
            f"  Pipe radius    : {self._pipe_radius} m\n"
            f"  Grid step      : {self._grid_step} (mesh downsample factor)\n"
            f"  Frame ID       : {self.FRAME_ID}"
        )

    # ---------------------------------------------------------------------- #
    # Waypoints callback — update stored robot path
    # ---------------------------------------------------------------------- #

    def _path_callback(self, msg: Path) -> None:
        self._waypoints = [
            (
                p.pose.position.x,
                p.pose.position.y,
                math.atan2(
                    2.0 * (p.pose.orientation.w * p.pose.orientation.z
                           + p.pose.orientation.x * p.pose.orientation.y),
                    1.0 - 2.0 * (p.pose.orientation.y ** 2
                                 + p.pose.orientation.z ** 2),
                ),
            )
            for p in msg.poses
        ]

    # ---------------------------------------------------------------------- #
    # Coverage callback
    # ---------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        """Decode heatmap, build TRIANGLE_LIST marker, publish."""
        try:
            heatmap = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge decode failed: {exc}")
            return

        marker = self._build_marker(heatmap, msg)
        self._publisher.publish(marker)

        self._frame_count += 1
        if self._frame_count % self.LOG_INTERVAL == 0:
            n_slices    = heatmap.shape[0]
            n_waypoints = len(self._waypoints)
            self.get_logger().info(
                f"Frame {self._frame_count:6d} | "
                f"Depth slices: {n_slices} | "
                f"Waypoints: {n_waypoints} | "
                f"Vertices: {len(marker.points)}"
            )

    # ---------------------------------------------------------------------- #
    # Marker construction
    # ---------------------------------------------------------------------- #

    def _build_marker(self, heatmap: np.ndarray, source_msg: Image) -> Marker:
        """
        Convert a mono8 heatmap (H × W) into a TRIANGLE_LIST Marker.

        H = number of accepted depth slices (grows only when robot moves).
        W = angle bins (360).

        Each grid cell (row i, col j) maps to one quad (two triangles).
        Triangle winding: CCW from outside → outward normals visible in RViz2.
        """
        marker = Marker()
        marker.header.stamp    = source_msg.header.stamp
        marker.header.frame_id = self.FRAME_ID
        marker.ns              = "pipe_coverage"
        marker.id              = 0
        marker.type            = Marker.TRIANGLE_LIST
        marker.action          = Marker.ADD
        marker.scale.x         = 1.0
        marker.scale.y         = 1.0
        marker.scale.z         = 1.0

        # Downsample to reduce triangle count
        step       = self._grid_step
        heatmap_ds = heatmap[::step, ::step]
        h, w       = heatmap_ds.shape

        if h < 2 or w < 2:
            return marker  # not enough grid cells yet

        coverage    = heatmap_ds.astype(np.float32) / 255.0
        vertex_grid = self._compute_vertex_grid(h, w)

        points: list[Point]     = []
        colors: list[ColorRGBA] = []

        for i in range(h - 1):
            for j in range(w):
                j_next = (j + 1) % w   # wrap the seam at 360°

                colour = self._coverage_to_color(float(coverage[i, j]))

                p_tl = self._make_point(vertex_grid[i,     j     ])
                p_bl = self._make_point(vertex_grid[i + 1, j     ])
                p_tr = self._make_point(vertex_grid[i,     j_next])
                p_br = self._make_point(vertex_grid[i + 1, j_next])

                # Triangle A: top-left → top-right → bottom-left (CCW from outside)
                points.extend([p_tl, p_tr, p_bl])
                colors.extend([colour, colour, colour])

                # Triangle B: bottom-left → top-right → bottom-right (CCW from outside)
                points.extend([p_bl, p_tr, p_br])
                colors.extend([colour, colour, colour])

        marker.points = points
        marker.colors  = colors
        return marker

    # ---------------------------------------------------------------------- #
    # Vertex grid computation
    # ---------------------------------------------------------------------- #

    def _compute_vertex_grid(self, h: int, w: int) -> np.ndarray:
        """Return (H, W, 3) float64 array of (x, y, z) world coordinates."""
        if len(self._waypoints) < 2:
            return self._compute_vertex_grid_fallback(h, w)
        return self._compute_vertex_grid_from_waypoints(h, w)

    def _compute_vertex_grid_fallback(self, h: int, w: int) -> np.ndarray:
        """
        Short straight cylinder along +X while waiting for waypoints.

        Shown at startup before the robot has moved far enough to accumulate
        two waypoints.  Disappears once real waypoints arrive.
        """
        angles  = np.linspace(0.0, 2.0 * math.pi, w, endpoint=False)
        x_vals  = np.linspace(0.0, self._pipe_radius * 2.0, h)
        a_grid  = np.tile(angles,         (h, 1))
        x_grid  = np.tile(x_vals[:, None], (1, w))
        ys = self._pipe_radius * np.cos(a_grid)
        zs = self._pipe_radius * np.sin(a_grid)
        return np.stack([x_grid, ys, zs], axis=-1)

    def _compute_vertex_grid_from_waypoints(self, h: int, w: int) -> np.ndarray:
        """
        Place cylinder cross-sections at the robot's actual odometry poses.

        For each downsampled row i, the corresponding waypoint is looked up
        by mapping back through the grid_step:
            wp_idx = min(i * grid_step, n_waypoints - 1)

        The cross-section at waypoint (x_i, y_i, yaw_i):
            e_v = (0, 0, 1)                    — world vertical
            e_h = (−sin(yaw), cos(yaw), 0)     — left of heading in XY
            P(θ) = (x_i, y_i, 0) + r·(cos(θ)·e_v + sin(θ)·e_h)

        This correctly handles straight sections, corners, and any arbitrary
        path the robot follows.
        """
        n_wp  = len(self._waypoints)
        step  = self._grid_step
        r     = self._pipe_radius
        theta = np.linspace(0.0, 2.0 * math.pi, w, endpoint=False)

        xs = np.zeros((h, w))
        ys = np.zeros((h, w))
        zs = np.zeros((h, w))

        for i in range(h):
            wp_idx       = min(i * step, n_wp - 1)
            x_i, y_i, yaw_i = self._waypoints[wp_idx]
            eh_x = -math.sin(yaw_i)
            eh_y =  math.cos(yaw_i)
            xs[i] = x_i + r * np.sin(theta) * eh_x
            ys[i] = y_i + r * np.sin(theta) * eh_y
            zs[i] =       r * np.cos(theta)

        return np.stack([xs, ys, zs], axis=-1)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _make_point(xyz: np.ndarray) -> Point:
        pt   = Point()
        pt.x = float(xyz[0])
        pt.y = float(xyz[1])
        pt.z = float(xyz[2])
        return pt

    @staticmethod
    def _coverage_to_color(cov: float) -> ColorRGBA:
        """
        Map a normalised coverage value [0, 1] to an RGBA colour.

        0.0       → dark blue   (unseen)
        0.0–0.1   → dark blue   (flat)
        0.1–0.5   → blue → cyan
        0.5–1.0   → cyan → red  (fully inspected)
        """
        colour   = ColorRGBA()
        colour.a = CylinderVisualizer3DNode._ALPHA

        if cov <= 0.1:
            colour.r, colour.g, colour.b = 0.2, 0.2, 0.8

        elif cov <= 0.5:
            t = (cov - 0.1) / 0.4
            colour.r, colour.g, colour.b = 0.2, 0.2 + t * 0.6, 0.8

        else:
            t = (cov - 0.5) / 0.5
            colour.r, colour.g, colour.b = 0.8 + t * 0.2, 0.2 - t * 0.2, 0.2

        return colour


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    node: CylinderVisualizer3DNode | None = None
    try:
        node = CylinderVisualizer3DNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
