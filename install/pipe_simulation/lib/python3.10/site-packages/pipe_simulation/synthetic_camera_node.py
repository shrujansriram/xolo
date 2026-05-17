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
  * any sphere obstacles (volumetric, centred in the cross-section)

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

Rays are unit vectors: |d|^2 = cos^2(theta) + sin^2(theta) = 1.

Cylinder intersection (camera inside the pipe, cam_z = 0):
  a   = d_y^2 + d_z^2
  b   = 2 * cam_y * d_y
  c   = cam_y^2 - R^2     (negative: camera always inside cylinder)
  t_cyl = (-b + sqrt(b^2 - 4ac)) / (2a)    [larger forward root]

Sphere intersection (ray-sphere, |d|=1 so a=1):
  oc   = cam_origin - sphere_centre
  b    = 2 * (d . oc)
  c    = |oc|^2 - r^2
  disc = b^2 - 4c
  t_sphere = (-b - sqrt(disc)) / 2   [smaller root = entry point]
  Hit when disc >= 0  AND  t_sphere > 0  AND  t_sphere < t_cyl

Key insight for coverage gaps:
  When the robot passes the sphere at (1.5, 0, 0) while at y=+0.08,
  rightward-looking rays (theta~90 deg, outer ring of fisheye) enter the
  sphere at t~0.04 m — much closer than the pipe wall at t~0.23 m.
  Those pixels fall in the outer ring that coverage_mapper samples.
  With threshold > 30, near-black [22,14,8] pixels are recorded as
  "uncovered" -> blue in the 3D cylinder.

Parameters
  pipe_radius       (default 0.15)  inner pipe radius [m]
  image_size        (default 320)   square image side [px]
  update_rate_hz    (default 10.0)  publish rate [Hz]
  use_sim_time      (default true)
  sphere_obstacles  (default "")    "x:radius" or "x:y:z:radius", comma-sep
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
        self.declare_parameter("sphere_obstacles", "")

        self._r    = float(self.get_parameter("pipe_radius").value)
        self._sz   = int(self.get_parameter("image_size").value)
        self._rate = float(self.get_parameter("update_rate_hz").value)

        # Parse sphere descriptors: "x:radius" or "x:y:z:radius", comma-sep.
        # Spheres are centred at (x, y, z) — default y=0, z=0 (pipe centre).
        spheres_str = str(self.get_parameter("sphere_obstacles").value).strip()
        self._spheres: list = []   # (cx, cy, cz, radius)
        if spheres_str:
            for part in spheres_str.split(","):
                tokens = [t.strip() for t in part.strip().split(":")]
                if len(tokens) == 2:
                    self._spheres.append((float(tokens[0]), 0.0, 0.0, float(tokens[1])))
                elif len(tokens) == 4:
                    self._spheres.append(tuple(float(t) for t in tokens))

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

        sphere_desc = (
            ", ".join(f"({sx:.2f},{sy:.2f},{sz:.2f}) r={sr:.3f}"
                      for sx, sy, sz, sr in self._spheres)
            or "none"
        )
        self.get_logger().info(
            f"synthetic_camera_node ready — {self._sz}x{self._sz} px "
            f"@ {self._rate} Hz — spheres: [{sphere_desc}] — physics ray-caster"
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

        # ---- Sphere obstacles (analytic ray-sphere intersection) -----------
        if self._spheres:
            img = self._render_spheres(img, cam_x, cam_y, d_x, d_y, d_z, t_cyl)

        return img

    # ------------------------------------------------------------------
    def _render_spheres(
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
        Paint sphere obstacles using analytic ray-sphere intersection.

        Rays are unit vectors (|d|=1), so the quadratic simplifies:
          a = 1
          b = 2 * (d . oc),   oc = cam_origin - sphere_centre
          c = |oc|^2 - r^2
          disc = b^2 - 4c
          t_entry = (-b - sqrt(disc)) / 2   [near intersection]

        When robot passes at y=+0.08 and sphere is at (1.5, 0, 0) r=0.04:
          rightward ray (theta=90°): oc=(0, 0.08, 0), b=2*(-1)*0.08=-0.16,
          c=0.08^2-0.04^2=0.0048, disc=0.0256-0.0192=0.0064>0.
          t_entry=(0.16-0.08)/2=0.04 m  << t_cyl~0.23 m  -> sphere blocks it.
        """
        for sx, sy, sz, sr in self._spheres:
            # oc = cam_origin - sphere_centre  (cam_z = 0)
            ocx = cam_x - sx
            ocy = cam_y - sy
            ocz = 0.0   - sz

            # a = 1 (unit rays)
            b    = 2.0 * (d_x * ocx + d_y * ocy + d_z * ocz)
            c    = ocx * ocx + ocy * ocy + ocz * ocz - sr * sr
            disc = b * b - 4.0 * c

            hit      = disc >= 0.0
            disc_s   = np.where(hit, disc, 0.0)
            t_sphere = np.where(hit, (-b - np.sqrt(disc_s)) / 2.0, 1e6)

            sphere_mask = (
                hit
                & (t_sphere > 1e-9)    # entry point is ahead of camera
                & (t_sphere < t_cyl)   # sphere is closer than pipe wall
                & self._valid          # inside fisheye circle
            )
            img[sphere_mask] = np.array([22, 14, 8], dtype=np.uint8)

        return img


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
