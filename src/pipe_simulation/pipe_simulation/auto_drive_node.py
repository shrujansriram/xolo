"""
auto_drive_node.py
------------------
Drives the pipe_bot through an L-shaped path:

    FORWARD_1  — travel leg1_m metres straight ahead
    PAUSE      — hold position for pause_s seconds
    TURN_LEFT  — rotate CCW by turn_deg degrees in place
    FORWARD_2  — travel leg2_m metres straight ahead
    DONE       — stop permanently

Parameters (ROS2)
  leg1_m       — first straight section length [m]     (default 2.0)
  pause_s      — stop duration between legs [s]        (default 2.0)
  turn_deg     — CCW turn angle in degrees             (default 90.0)
  leg2_m       — second straight section length [m]    (default 2.0)
  drive_speed  — forward speed [m/s]                   (default 0.10)
  turn_speed   — angular speed [rad/s]                 (default 0.40)
  cmd_rate_hz  — cmd_vel publish rate [Hz]             (default 10.0)
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

# State machine states
_INIT      = "INIT"
_FORWARD_1 = "FORWARD_1"
_PAUSE     = "PAUSE"
_TURN_LEFT = "TURN_LEFT"
_FORWARD_2 = "FORWARD_2"
_DONE      = "DONE"


class AutoDrive(Node):
    def __init__(self):
        super().__init__("auto_drive_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("leg1_m",       2.0)
        self.declare_parameter("pause_s",      2.0)
        self.declare_parameter("turn_deg",    90.0)
        self.declare_parameter("leg2_m",       2.0)
        self.declare_parameter("drive_speed",  0.10)
        self.declare_parameter("turn_speed",   0.40)
        self.declare_parameter("cmd_rate_hz", 10.0)

        self._leg1_m      = float(self.get_parameter("leg1_m").value)
        self._pause_s     = float(self.get_parameter("pause_s").value)
        self._turn_rad    = math.radians(float(self.get_parameter("turn_deg").value))
        self._leg2_m      = float(self.get_parameter("leg2_m").value)
        self._drive_speed = float(self.get_parameter("drive_speed").value)
        self._turn_speed  = float(self.get_parameter("turn_speed").value)
        self._rate_hz     = float(self.get_parameter("cmd_rate_hz").value)

        # ------------------------------------------------------------------ #
        # State machine
        # ------------------------------------------------------------------ #
        self._state: str = _INIT

        # Odometry tracking
        self._last_x:    float | None = None
        self._last_y:    float | None = None
        self._yaw:       float        = 0.0   # current heading [rad]
        self._distance:  float        = 0.0   # accumulated travel in current phase

        # Phase bookkeeping
        self._pause_start:    float | None = None   # wall-clock time when pause began
        self._yaw_turn_start: float | None = None   # heading at start of turn

        # ------------------------------------------------------------------ #
        # ROS2 I/O
        # ------------------------------------------------------------------ #
        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self._timer = self.create_timer(
            1.0 / self._rate_hz, self._publish_cmd
        )

        self.get_logger().info(
            f"auto_drive_node ready — L-path: "
            f"forward {self._leg1_m} m → pause {self._pause_s} s → "
            f"turn {math.degrees(self._turn_rad):.0f}° → "
            f"forward {self._leg2_m} m"
        )

    # ---------------------------------------------------------------------- #
    # Odometry callback — maintain position + accumulated distance + yaw
    # ---------------------------------------------------------------------- #

    def _odom_callback(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        if self._state == _INIT:
            self._last_x = x
            self._last_y = y
            self._state  = _FORWARD_1
            self.get_logger().info(
                f"Odometry ready — starting FORWARD_1 ({self._leg1_m} m)"
            )
            return

        if self._last_x is None:
            return

        step = math.hypot(x - self._last_x, y - self._last_y)
        self._distance += step
        self._last_x    = x
        self._last_y    = y

    # ---------------------------------------------------------------------- #
    # Timer callback — state machine logic
    # ---------------------------------------------------------------------- #

    def _publish_cmd(self) -> None:
        cmd = Twist()   # default: zero velocity

        if self._state == _INIT:
            # Waiting for first odometry message — publish zero and wait
            self._pub.publish(cmd)
            return

        if self._state == _FORWARD_1:
            if self._distance >= self._leg1_m:
                self._distance      = 0.0
                self._pause_start   = time.monotonic()
                self._state         = _PAUSE
                self.get_logger().info(
                    f"Leg 1 complete ({self._leg1_m} m) — pausing "
                    f"{self._pause_s} s"
                )
            else:
                cmd.linear.x = self._drive_speed

        elif self._state == _PAUSE:
            elapsed = time.monotonic() - self._pause_start
            if elapsed >= self._pause_s:
                self._yaw_turn_start = self._yaw
                self._state          = _TURN_LEFT
                self.get_logger().info(
                    f"Pause complete — turning "
                    f"{math.degrees(self._turn_rad):.0f}° CCW"
                )
            # cmd stays zero (hold position)

        elif self._state == _TURN_LEFT:
            # Wrap-safe angular displacement
            delta = math.atan2(
                math.sin(self._yaw - self._yaw_turn_start),
                math.cos(self._yaw - self._yaw_turn_start),
            )
            if abs(delta) >= self._turn_rad:
                self._distance = 0.0
                self._state    = _FORWARD_2
                self.get_logger().info(
                    f"Turn complete — starting FORWARD_2 ({self._leg2_m} m)"
                )
            else:
                cmd.angular.z = self._turn_speed

        elif self._state == _FORWARD_2:
            if self._distance >= self._leg2_m:
                self._state = _DONE
                self.get_logger().info(
                    f"Leg 2 complete ({self._leg2_m} m) — stopped."
                )
            else:
                cmd.linear.x = self._drive_speed

        # _DONE: cmd stays zero — robot is permanently stopped

        self._pub.publish(cmd)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node: AutoDrive | None = None
    try:
        node = AutoDrive()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
