"""
cylinder_visualizer_3d_node.py
-------------------------------
ROS2 node that converts the growing 2D coverage heatmap into a 3D
cylindrical TRIANGLE_LIST Marker for display in RViz2.

The heatmap published by CoverageMapperNode has:
  - Height = number of frames processed (grows in real-time)
  - Width  = 360 (angle bins)

Each row represents one camera depth position.  The cylinder length is
computed automatically as:

    pipe_length = heatmap_height * depth_per_frame

so the cylinder grows longer in real-time as the camera advances through
the pipe.  No pipe_length parameter is needed.

Colour mapping:
    cov = 0  →  dark blue   (not yet inspected)
    cov = 1  →  red         (inspected)

Subscriptions
-------------
    /pipe/coverage              (sensor_msgs/Image)         — mono8 heatmap

Publications
------------
    /visualization/cylinder     (visualization_msgs/Marker) — TRIANGLE_LIST

Usage (after building the package):
    ros2 run pipe_perception cylinder_visualizer_3d_node
    ros2 run pipe_perception cylinder_visualizer_3d_node --ros-args \
        -p pipe_radius:=0.15 -p depth_per_frame:=0.01
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class CylinderVisualizer3DNode(Node):
    """
    Rebuilds the 3D cylinder marker on every coverage heatmap update.

    The cylinder length grows dynamically with the number of frames processed.
    pipe_radius is the only geometry parameter required at launch.

    Parameters (ROS2)
    -----------------
    pipe_radius : float
        Cylinder radius in metres.  Default: 0.15
    depth_per_frame : float
        Physical distance (metres) the camera advances between consecutive
        frames.  Controls how quickly the cylinder grows in length.
        Default: 0.01  (1 cm/frame → 0.3 m/s at 30 fps)
    grid_step : int
        Downsample factor applied to the heatmap before building the mesh.
        grid_step=1 → full resolution (very slow); grid_step=4 → 16× fewer
        triangles, good balance of speed and detail.  Default: 4
    """

    SUB_TOPIC    = "/pipe/coverage"
    PUB_TOPIC    = "/visualization/cylinder"
    FRAME_ID     = "world"
    LOG_INTERVAL = 10  # frames between status log lines

    _ALPHA = 1.0   # fully opaque

    def __init__(self) -> None:
        super().__init__("cylinder_visualizer_3d_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("pipe_radius",      0.15)
        self.declare_parameter("depth_per_frame",  0.01)
        self.declare_parameter("grid_step",        4)

        self._pipe_radius: float    = float(self.get_parameter("pipe_radius").value)
        self._depth_per_frame: float = float(self.get_parameter("depth_per_frame").value)
        self._grid_step: int        = max(1, int(self.get_parameter("grid_step").value))

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._bridge       = CvBridge()
        self._frame_count  = 0

        # ------------------------------------------------------------------ #
        # Parameter update callback (allows live depth_per_frame changes)
        # ------------------------------------------------------------------ #
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # ------------------------------------------------------------------ #
        # Subscriber / Publisher
        # ------------------------------------------------------------------ #
        self._subscriber = self.create_subscription(
            Image,
            self.SUB_TOPIC,
            self._image_callback,
            qos_profile=10,
        )
        self._publisher = self.create_publisher(
            Marker, self.PUB_TOPIC, qos_profile=10
        )

        # ------------------------------------------------------------------ #
        # Startup log
        # ------------------------------------------------------------------ #
        self.get_logger().info(
            f"CylinderVisualizer3DNode ready.\n"
            f"  Subscribing    : {self.SUB_TOPIC}\n"
            f"  Publishing     : {self.PUB_TOPIC}\n"
            f"  Pipe radius    : {self._pipe_radius} m\n"
            f"  Depth/frame    : {self._depth_per_frame} m  "
            f"(cylinder grows in real-time)\n"
            f"  Grid step      : {self._grid_step} (mesh downsample factor)\n"
            f"  Frame ID       : {self.FRAME_ID}"
        )

    # ---------------------------------------------------------------------- #
    # Parameter callback — allows odom_depth_node to push live updates
    # ---------------------------------------------------------------------- #

    def _on_parameters_changed(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "depth_per_frame" and float(param.value) > 0.0:
                self._depth_per_frame = float(param.value)
                self.get_logger().info(
                    f"depth_per_frame updated to {self._depth_per_frame:.5f} m/frame"
                )
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------------- #
    # Callback
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
            n_frames   = heatmap.shape[0]
            pipe_length = n_frames * self._depth_per_frame
            self.get_logger().info(
                f"Frame {self._frame_count:6d} | "
                f"Depth slices: {n_frames} | "
                f"Pipe length: {pipe_length:.3f} m | "
                f"Vertices: {len(marker.points)}"
            )

    # ---------------------------------------------------------------------- #
    # Marker construction
    # ---------------------------------------------------------------------- #

    def _build_marker(self, heatmap: np.ndarray, source_msg: Image) -> Marker:
        """
        Convert a mono8 heatmap (H x W) into a TRIANGLE_LIST Marker.

        H = number of depth slices (frames) processed so far.
        W = angle bins (360).
        pipe_length is computed as H * depth_per_frame.

        The cylinder is oriented along the +Z axis, starting at the world
        origin.  Each grid cell (row i, col j) maps to one quad (two
        triangles) on the cylinder surface.

        Triangle winding order: CCW when viewed from outside → outward
        normals → front faces visible in RViz2.
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

        # Normalise to [0.0, 1.0]
        coverage = heatmap_ds.astype(np.float32) / 255.0

        # Physical pipe length grows with depth slices
        # Original (un-downsampled) height × metres-per-frame = total length
        pipe_length = heatmap.shape[0] * self._depth_per_frame

        vertex_grid = self._compute_vertex_grid(h, w, pipe_length)

        points: list[Point]    = []
        colors: list[ColorRGBA] = []

        for i in range(h - 1):
            for j in range(w):
                j_next = (j + 1) % w          # wrap the seam at 360°

                cov_value: float    = float(coverage[i, j])
                colour: ColorRGBA   = self._coverage_to_color(cov_value)

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

    def _compute_vertex_grid(
        self, h: int, w: int, pipe_length: float
    ) -> np.ndarray:
        """
        Return a (H, W, 3) float64 array of (x, y, z) cylinder coordinates.

        Rows → evenly spaced Z positions [0, pipe_length].
        Cols → evenly spaced angles [0, 2π) around the circumference.

        pipe_length is passed explicitly so the caller can compute it from
        the current number of depth slices × depth_per_frame.
        """
        angles = np.linspace(0.0, 2.0 * math.pi, w, endpoint=False)
        z_vals = np.linspace(0.0, pipe_length, h)

        angle_grid = np.tile(angles,         (h, 1))  # (H, W)
        z_grid     = np.tile(z_vals[:, None], (1, w))  # (H, W)

        xs = self._pipe_radius * np.cos(angle_grid)
        ys = self._pipe_radius * np.sin(angle_grid)

        return np.stack([xs, ys, z_grid], axis=-1)

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
