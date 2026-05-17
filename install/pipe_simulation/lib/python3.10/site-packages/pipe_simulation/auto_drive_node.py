"""
auto_drive_node.py
------------------
Drives the pipe_bot through a straight pipe while smoothly manoeuvring
around half-disc wall obstacles.

Physics rationale
-----------------
Each wall blocks exactly half the pipe cross-section (y > 0 for "left" walls,
y < 0 for "right" walls).  The robot must swing laterally to the clear half
before reaching the wall, pass through, then return to the centreline.

A proportional Y-tracking controller handles this continuously:

  y_desired(x) = Σ  sign_i · y_offset · bell(x, x_wall_i, avoid_range)

where bell is a raised-cosine bump centred on x_wall_i:
  bell(x) = 0.5 · (1 + cos(π · (x − x_wall) / avoid_range))
             for |x − x_wall| < avoid_range, else 0

  sign = −1 for a left-blocking wall (robot dodges right, y < 0)
  sign = +1 for a right-blocking wall (robot dodges left, y > 0)

The angular command is:
  ω = clamp(Kp · (y_desired − y_actual),  −ω_max, +ω_max)

State machine:  INIT → DRIVING → DONE

Parameters (ROS2)
  pipe_length  — robot drives until x ≥ pipe_length − 0.10 m   (default 3.0)
  drive_speed  — forward speed [m/s]                             (default 0.10)
  y_offset     — peak lateral offset to clear the wall edge [m]  (default 0.07)
  avoid_range  — half-width of avoidance bell around each wall   (default 0.35)
  kp_y         — proportional gain for lateral tracking          (default 6.0)
  omega_max    — maximum angular velocity [rad/s]                (default 0.40)
  cmd_rate_hz  — publish rate [Hz]                               (default 10.0)
  walls        — comma-separated "x:side" descriptors
                 e.g. "0.2:left,1.2:left,1.9:right,2.6:right"
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

_INIT    = "INIT"
_DRIVING = "DRIVING"
_DONE    = "DONE"


class AutoDrive(Node):
    def __init__(self):
        super().__init__("auto_drive_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("pipe_length",  3.0)
        self.declare_parameter("drive_speed",  0.10)
        self.declare_parameter("y_offset",     0.07)
        self.declare_parameter("avoid_range",  0.35)
        self.declare_parameter("kp_y",         6.0)
        self.declare_parameter("omega_max",    0.40)
        self.declare_parameter("cmd_rate_hz",  10.0)
        self.declare_parameter("walls",        "")

        self._pipe_length = float(self.get_parameter("pipe_length").value)
        self._drive_speed = float(self.get_parameter("drive_speed").value)
        self._y_offset    = float(self.get_parameter("y_offset").value)
        self._avoid_range = float(self.get_parameter("avoid_range").value)
        self._kp_y        = float(self.get_parameter("kp_y").value)
        self._omega_max   = float(self.get_parameter("omega_max").value)
        self._rate_hz     = float(self.get_parameter("cmd_rate_hz").value)

        # Parse wall descriptors: "x:side,..."
        # sign = −1 for left walls (dodge right, y<0)
        # sign = +1 for right walls (dodge left, y>0)
        walls_str = str(self.get_parameter("walls").value).strip()
        self._wall_info: list = []   # (x_wall, sign)
        if walls_str:
            for part in walls_str.split(","):
                part = part.strip()
                if ":" in part:
                    x_str, side = part.split(":", 1)
                    sign = -1.0 if side.strip() == "left" else +1.0
                    self._wall_info.append((float(x_str.strip()), sign))

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._state   = _INIT
        self._robot_x = 0.0
        self._robot_y = 0.0

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

        walls_desc = (
            ", ".join(f"x={x:.2f}({'L' if s < 0 else 'R'})" for x, s in self._wall_info)
            or "none"
        )
        self.get_logger().info(
            f"auto_drive_node ready — straight drive {self._pipe_length - 0.10:.2f} m, "
            f"walls: [{walls_desc}], "
            f"y_offset={self._y_offset} m, avoid_range={self._avoid_range} m"
        )

    # ---------------------------------------------------------------------- #
    # Odometry callback
    # ---------------------------------------------------------------------- #

    def _odom_callback(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

        if self._state == _INIT:
            self._state = _DRIVING
            self.get_logger().info(
                f"Odometry received — DRIVING (stopping at x≥"
                f"{self._pipe_length - 0.10:.2f} m)"
            )

    # ---------------------------------------------------------------------- #
    # Desired Y trajectory
    # ---------------------------------------------------------------------- #

    def _desired_y(self, x: float) -> float:
        """
        Target lateral position (metres) at pipe-axis position x.

        Each wall contributes a raised-cosine bell offset centred on
        x_wall with half-width avoid_range.  Left walls drive the robot
        to y < 0 (clear the +Y blockage); right walls to y > 0.

        Overlapping bells sum — walls are positioned to avoid overlap.
        """
        y_des = 0.0
        for x_wall, sign in self._wall_info:
            t = (x - x_wall) / self._avoid_range
            if abs(t) < 1.0:
                bell = 0.5 * (1.0 + math.cos(math.pi * t))
                y_des += sign * self._y_offset * bell
        return y_des

    # ---------------------------------------------------------------------- #
    # Timer callback — publish velocity commands
    # ---------------------------------------------------------------------- #

    def _publish_cmd(self) -> None:
        cmd = Twist()   # default: zero velocity

        if self._state == _INIT:
            self._pub.publish(cmd)
            return

        if self._state == _DRIVING:
            stop_x = self._pipe_length - 0.10
            if self._robot_x >= stop_x:
                self._state = _DONE
                self.get_logger().info(
                    f"Pipe end reached (x={self._robot_x:.3f} m) — DONE."
                )
            else:
                y_des    = self._desired_y(self._robot_x)
                y_err    = y_des - self._robot_y
                omega    = self._kp_y * y_err
                omega    = max(-self._omega_max, min(self._omega_max, omega))

                cmd.linear.x  = self._drive_speed
                cmd.angular.z = omega

        # _DONE: cmd stays zero — robot permanently stopped

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
