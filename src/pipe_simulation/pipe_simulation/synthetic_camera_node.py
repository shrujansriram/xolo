"""
synthetic_camera_node.py
------------------------
Publishes synthetic fisheye images of a pipe interior on /camera/image_raw.

Physics-based ray-caster — fully position and heading aware.

Camera model: equidistant fisheye projection
  theta = R_norm * (pi/2)   R_norm in [0,1] -> theta in [0, pi/2]

For each fisheye pixel a ray is cast from the robot's current (x, y, yaw)
into the 3D world.  The pipe is a cylinder of radius pipe_radius aligned
with the +X axis.  Rays are intersected analytically with:
  * the pipe cylinder surface
  * any rock obstacles (volumetric axis-aligned boxes)

Coordinate conventions
-----------------------
  World  : +X along pipe, +Y left of robot, +Z up
  Image  : u increases rightward (world -Y at yaw=0)
           v increases downward  (world -Z)
  phi    : atan2(v, u) in image plane

Camera-frame unit vectors (in world frame, given robot heading yaw):
  fwd = ( cos yaw,  sin yaw,  0 )   optical axis
  rgt = ( sin yaw, -cos yaw,  0 )   image right = world -Y at yaw=0
  dn  = ( 0,        0,       -1 )   image down

Ray direction for pixel (cos_theta, sin_theta, cos_phi, sin_phi):
  d_x =  cos_theta * cos_yaw + sin_theta * cos_phi * sin_yaw
  d_y =  cos_theta * sin_yaw - sin_theta * cos_phi * cos_yaw
  d_z = -sin_theta * sin_phi

Cylinder intersection (camera inside the pipe, cam_z = 0):
  a   = d_y^2 + d_z^2
  b   = 2 * cam_y * d_y
  c   = cam_y^2 - R^2     (negative: camera always inside cylinder)
  t_cyl = (-b + sqrt(b^2 - 4ac)) / (2a)    [larger forward root]

Rock intersection (AABB slab test):
  The rock is a volumetric box [x1,x2] x [y1,y2] x [z1,z2].
  Standard slab intersection: compute per-axis (t_enter, t_exit) intervals,
  then t_enter_total = max of t_enters, t_exit_total = min of t_exits.
  A hit occurs when t_enter_total < t_exit_total and t_exit_total > 0.
  If t_enter_total < t_cyl, the rock occludes the pipe wall -> near-black.

Key insight for coverage gaps:
  When the robot is INSIDE the rock's x-range [x1, x2], sideways-looking
  rays (theta~90 deg, outer ring of the fisheye) hit the rock at a short
  distance before the pipe wall.  Those pixels are in the outer ring of the
  cylindrical-unwrapped image that the coverage mapper samples.  With the
  threshold fixed to >30 in coverage_mapper_node, near-black [22,14,8]
  pixels are recorded as "uncovered" -> blue in the 3D cylinder.

Parameters
  pipe_radius      (default 0.15)  inner pipe radius [m]
  image_size       (default 320)   square image side [px]
  update_rate_hz   (default 10.0)  publish rate [Hz]
  use_sim_time     (default true)
  walls            (default "")    e.g. "1.5:right"
  rock_half_length (default 0.20)  half-length of rock box in X [m]
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

        self.declare_parameter("pipe_radius",      0.15)
        self.declare_parameter("image_size",       320)
        self.declare_parameter("update_rate_hz",   10.0)
        self.declare_parameter("walls",            "")
        self.declare_parameter("rock_half_length", 0.20)

        self._r    = float(self.get_parameter("pipe_radius").value)
        self._sz   = int(self.get_parameter("image_size").value)
        self._rate = float(self.get_parameter("update_rate_hz").value)
        self._rhl  = float(self.get_parameter("rock_half_length").value)

        # Parse wall/rock descriptors: "x:side,x:side,..."
        walls_str = str(self.get_parameter("walls").value).strip()
        self._rocks: list = []
        if walls_str:
            for part in walls_str.split(","):
                part = part.strip()
                if ":" in part:
                    x_str, side = part.split(":", 1)
                    self._rocks.append((float(x_str.strip()), side.strip()))

        # Robot pose — updated from /odom
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0

        self._bridge = CvBridge()

        # Static pixel-geometry LUTs (independent of robot pose)
        (self._cos_theta,
         self._sin_theta,
         self._cos_phi,
         self._sin_phi,
         self._valid) = self._build_lut(self._sz)

        self._pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self._sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )
        self._timer = self.create_timer(
            1.0 / self._rate, self._publish_frame
        )

        rock_desc = (
            ", ".join(f"x={x:.2f}:{s}" for x, s in self._rocks)
            or "none"
        )
        self.get_logger().info(
            f"synthetic_camera_node ready — {self._sz}x{self._sz} px "
            f"@ {self._rate} Hz — rocks: [{rock_desc}] "
            f"half_length={self._rhl:.2f} m — physics ray-caster"
        )

    # ------------------------------------------------------------------
    # Static pixel-geometry LUTs
    # ------------------------------------------------------------------
    @staticmethod
    def _build_lut(sz: int):
        cx = cy = sz / 2.0
        u_1d = (np.arange(sz) - cx) / cx   # -1 to +1 (right)
        v_1d = (np.arange(sz) - cy) / cy   # -1 to +1 (down)
        U, V = np.meshgrid(u_1d, v_1d)

        R_norm = np.sqrt(U * U + V * V)
        theta  = np.clip(R_norm, 0.0, 1.0) * (math.pi / 2.0)

        cos_theta = np.cos(theta).astype(np.float32)
        sin_theta = np.sin(theta).astype(np.float32)

        phi     = np.arctan2(V, U)
        cos_phi = np.cos(phi).astype(np.float32)
        sin_phi = np.sin(phi).astype(np.float32)

        valid = (R_norm <= 1.0)
        return cos_theta, sin_theta, cos_phi, sin_phi, valid

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    # ------------------------------------------------------------------
    def _publish_frame(self):
        img = self._render(self._robot_x, self._robot_y, self._robot_yaw)
        ros_img = self._bridge.cv2_to_imgmsg(img, encoding="rgb8")
        ros_img.header.stamp    = self.get_clock().now().to_msg()
        ros_img.header.frame_id = "camera_link"
        self._pub.publish(ros_img)

    # ------------------------------------------------------------------
    def _render(self, cam_x: float, cam_y: float, yaw: float) -> np.ndarray:
        """Ray-cast from camera pose into the pipe world."""
        R       = self._r
        cos_t   = self._cos_theta
        sin_t   = self._sin_theta
        cos_p   = self._cos_phi
        sin_p   = self._sin_phi
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # ---- Ray directions in world frame --------------------------------
        # fwd=(cos_yaw, sin_yaw, 0), rgt=(sin_yaw,-cos_yaw,0), dn=(0,0,-1)
        d_x = cos_t * cos_yaw + sin_t * cos_p * sin_yaw
        d_y = cos_t * sin_yaw - sin_t * cos_p * cos_yaw
        d_z = -sin_t * sin_p

        # ---- Cylinder intersection ----------------------------------------
        # Pipe: y^2 + z^2 = R^2, cam at (cam_x, cam_y, 0)
        a    = d_y * d_y + d_z * d_z
        b    = 2.0 * cam_y * d_y
        c_sc = cam_y * cam_y - R * R
        disc = b * b - 4.0 * a * c_sc

        _EPS   = 1e-9
        a_safe = np.where(a < _EPS, 1.0, a)
        t_cyl  = np.where(
            a < _EPS,
            1e6,
            (-b + np.sqrt(np.clip(disc, 0.0, None))) / (2.0 * a_safe),
        )
        t_cyl = np.maximum(t_cyl, 0.0)

        # ---- Cylinder hit point and texture --------------------------------
        hit_x    = cam_x + t_cyl * d_x
        hit_y    = cam_y + t_cyl * d_y
        hit_z    =          t_cyl * d_z
        phi_pipe = np.arctan2(hit_z, hit_y)

        texture = (
            0.75
            + 0.15 * np.sin(phi_pipe * 8.0)
            + 0.10 * np.sin(phi_pipe * 3.0 + hit_x * 15.0)
        )
        n_y        = hit_y / R
        n_z        = hit_z / R
        cos_inc    = np.abs(d_y * n_y + d_z * n_z)
        depth_fade = np.exp(-np.clip(t_cyl - R, 0.0, None) * 0.35)
        shade      = np.clip(depth_fade * (0.3 + 0.7 * cos_inc), 0.0, 1.0)

        lum = np.clip(shade * texture * 180.0 + 20.0, 10.0, 240.0)
        ch  = lum.astype(np.uint8)
        img = np.stack([ch, ch, ch], axis=-1)

        img[~self._valid] = 0

        # ---- Rock obstacles (volumetric AABB slab intersection) -----------
        if self._rocks:
            img = self._render_rocks(img, cam_x, cam_y, d_x, d_y, d_z, t_cyl)

        return img

    # ------------------------------------------------------------------
    def _slab(
        self,
        d: np.ndarray,
        origin: float,
        lo: float,
        hi: float,
    ):
        """
        Per-axis slab intersection for AABB ray-box test (vectorised).

        Returns (t_lo, t_hi) arrays representing the [entry, exit] parameter
        range for this axis.

        When d=0 (ray parallel to slab):
          * origin inside [lo, hi]  -> entire ray is inside: t in (-inf, +inf)
          * origin outside [lo, hi] -> no intersection:      t in (+inf, -inf)
        """
        _E   = 1e-9
        ok   = np.abs(d) > _E
        # Safe denominator (avoid /0 — result ignored when not ok)
        d_s  = np.where(ok, d, 1.0)
        t_a  = np.where(ok, (lo - origin) / d_s, 0.0)
        t_b  = np.where(ok, (hi - origin) / d_s, 0.0)

        inside = (origin >= lo) and (origin <= hi)
        t_lo = np.where(ok, np.minimum(t_a, t_b), (-1e7 if inside else  1e7))
        t_hi = np.where(ok, np.maximum(t_a, t_b), ( 1e7 if inside else -1e7))
        return t_lo, t_hi

    def _render_rocks(
        self,
        img:   np.ndarray,
        cam_x: float,
        cam_y: float,
        d_x:   np.ndarray,
        d_y:   np.ndarray,
        d_z:   np.ndarray,
        t_cyl: np.ndarray,
    ) -> np.ndarray:
        """
        Paint rock obstacles using ray-AABB (slab) intersection.

        Each rock is a volumetric box:
          X: [x_wall - rock_half_length, x_wall + rock_half_length]
          Y: [-R, -0.01]  for 'right' rock  (right half, 1 cm gap)
             [+0.01,  R]  for 'left'  rock
          Z: [-R, R]       full pipe height

        Critical physics: when the robot is INSIDE the rock's X-range,
        sideways rays (theta~90 deg, phi~0) hit the rock at a very short
        distance before the pipe wall.  These pixels appear in the OUTER
        ring of the cylindrical-unwrapped image — the ring that the
        coverage_mapper_node samples for angular coverage.  Painting them
        near-black [22, 14, 8] causes the coverage mapper (threshold > 30)
        to record those angular bins as 'uncovered', producing a blue strip
        in the 3-D cylinder visualization.
        """
        R  = self._r
        hl = self._rhl

        for x_wall, side in self._rocks:
            x1, x2 = x_wall - hl, x_wall + hl
            if side == "right":
                y1, y2 = -R, -0.01   # right half of pipe cross-section
            else:
                y1, y2 = 0.01, R     # left half
            z1, z2 = -R, R

            tx_lo, tx_hi = self._slab(d_x, cam_x, x1, x2)
            ty_lo, ty_hi = self._slab(d_y, cam_y, y1, y2)
            tz_lo, tz_hi = self._slab(d_z, 0.0,   z1, z2)

            t_enter = np.maximum(np.maximum(tx_lo, ty_lo), tz_lo)
            t_exit  = np.minimum(np.minimum(tx_hi, ty_hi), tz_hi)

            rock_mask = (
                (t_enter < t_exit)        # valid box intersection
                & (t_exit  > 1e-9)        # box is (at least partly) ahead
                & (t_enter < t_cyl)       # box is closer than pipe wall
                & self._valid             # inside fisheye circle
            )
            img[rock_mask] = np.array([22, 14, 8], dtype=np.uint8)

        return img


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
