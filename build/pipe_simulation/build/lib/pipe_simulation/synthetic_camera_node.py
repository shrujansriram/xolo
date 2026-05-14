"""
synthetic_camera_node.py
------------------------
Publishes synthetic fisheye images of a pipe interior on /camera/image_raw.

Bypasses Gazebo's Ogre2 renderer (which crashes on WSL2 software EGL due to
the Default/TransGreen fixed-function material bug).

The equidistant fisheye projection is reproduced analytically:
  r_px / r_max  =  theta / (pi/2)
  where theta is the angle from the optical axis (0 = straight ahead,
  pi/2 = directly at the pipe wall).

The pipe wall appears at exactly 50 % of the image radius when the camera
is at the pipe centreline, matching outer_fraction=0.50 in coverage_mapper.

Parameters
  pipe_radius    (default 0.15) — inner pipe radius [m]
  image_size     (default 320)  — square image side length [px]
  update_rate_hz (default 10.0) — publish frequency [Hz]
  use_sim_time   (default true)
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image


class SyntheticCamera(Node):
    def __init__(self):
        super().__init__("synthetic_camera_node")

        self.declare_parameter("pipe_radius",    0.15)
        self.declare_parameter("image_size",     320)
        self.declare_parameter("update_rate_hz", 10.0)

        self._r     = self.get_parameter("pipe_radius").value
        self._sz    = self.get_parameter("image_size").value
        self._rate  = self.get_parameter("update_rate_hz").value

        self._robot_x = 0.0
        self._bridge  = CvBridge()

        # Pre-compute static lookup tables for the fisheye projection
        self._lut_shade, self._lut_phi = self._build_lut(self._sz)

        self._pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self._sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )
        self._timer = self.create_timer(
            1.0 / self._rate, self._publish_frame
        )
        self.get_logger().info(
            f"synthetic_camera_node ready — {self._sz}×{self._sz} px "
            f"@ {self._rate} Hz"
        )

    # ------------------------------------------------------------------
    # Build static LUTs  (only called once at startup)
    # ------------------------------------------------------------------
    def _build_lut(self, sz: int):
        cx = cy = sz / 2.0
        # Coordinate grids
        xs = (np.arange(sz) - cx) / cx          # [-1 … 1]
        ys = (np.arange(sz) - cy) / cy
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X * X + Y * Y)              # normalised radius

        # Equidistant fisheye: r = f*theta  →  theta = r*(pi/2)
        theta = np.clip(R, 0.0, 1.0) * (math.pi / 2.0)

        # Shading: looking at the wall (theta≈pi/2) → bright;
        #          looking forward down the pipe (theta≈0) → dark
        shade = np.sin(theta)                    # 0 … 1

        # Azimuth angle for texture stripe pattern
        phi = np.arctan2(Y, X)                  # -pi … pi

        # Mask pixels outside the circular fisheye area
        mask = R > 1.0
        shade[mask] = -1.0                       # sentinel for black

        return shade.astype(np.float32), phi.astype(np.float32)

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x

    # ------------------------------------------------------------------
    def _publish_frame(self):
        img = self._render(self._robot_x)

        ros_img = self._bridge.cv2_to_imgmsg(img, encoding="rgb8")
        ros_img.header.stamp    = self.get_clock().now().to_msg()
        ros_img.header.frame_id = "camera_link"
        self._pub.publish(ros_img)

    # ------------------------------------------------------------------
    def _render(self, robot_x: float) -> np.ndarray:
        sz = self._sz
        shade = self._lut_shade
        phi   = self._lut_phi

        # Concrete-grey pipe wall texture (longitudinal + azimuthal stripes)
        # Phase shifts with robot_x so new texture appears as robot moves
        texture = (
            0.75
            + 0.15 * np.sin(phi * 8.0)                       # azimuth stripes
            + 0.10 * np.sin(phi * 3.0 + robot_x * 15.0)      # slow rotation
        )

        # Base luminance: wall bright, pipe-ahead dark
        lum = np.clip(shade * texture * 180.0 + 20.0, 10.0, 240.0)

        # Build RGB (grey concrete)
        ch = lum.astype(np.uint8)
        img = np.stack([ch, ch, ch], axis=-1)

        # Black for pixels outside the fisheye circle
        outside = shade < 0.0
        img[outside] = 0

        return img


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
