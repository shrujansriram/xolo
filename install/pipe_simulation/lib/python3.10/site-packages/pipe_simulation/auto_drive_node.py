"""
auto_drive_node.py
------------------
Drives the pipe_bot through the pipe with a variable-speed profile so that
some sections are traversed quickly (producing longitudinal coverage gaps)
while others are traversed slowly (producing dense coverage).

Speed profile  (distance from start → speed)
  0.0 – 1.2 m  :  0.06 m/s  ← slow, dense coverage
  1.2 – 2.0 m  :  0.35 m/s  ← fast burst  → GAP A  (missed section)
  2.0 – 2.8 m  :  0.06 m/s  ← slow, dense coverage
  2.8 – 3.4 m  :  0.28 m/s  ← fast burst  → GAP B  (missed section)
  3.4 – 4.5 m  :  0.06 m/s  ← slow, dense coverage

Parameters (all ROS2 parameters, settable from launch or command line)
  pipe_length    (default 4.50)  — metres to travel before stopping
  cmd_rate_hz    (default 10.0)  — rate at which cmd_vel is published
  use_sim_time   (default true)  — must match the rest of the stack
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# (distance_start, distance_end, speed_m_s)
_SPEED_PROFILE = [
    (0.0,  1.2,  0.06),   # slow — dense coverage
    (1.2,  2.0,  0.35),   # fast burst — GAP A
    (2.0,  2.8,  0.06),   # slow — dense coverage
    (2.8,  3.4,  0.28),   # fast burst — GAP B
    (3.4,  4.5,  0.06),   # slow — dense coverage
]


def _speed_at(distance: float) -> float:
    """Return the desired forward speed for the current odometry distance."""
    for d_start, d_end, spd in _SPEED_PROFILE:
        if distance < d_end:
            return spd
    return 0.0   # past the last segment


class AutoDrive(Node):
    def __init__(self):
        super().__init__("auto_drive_node")

        self.declare_parameter("pipe_length",  4.50)
        self.declare_parameter("cmd_rate_hz",  10.0)

        self._pipe_len   = self.get_parameter("pipe_length").value
        self._rate_hz    = self.get_parameter("cmd_rate_hz").value

        self._start_x    = None
        self._distance   = 0.0
        self._done       = False

        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )
        self._timer = self.create_timer(
            1.0 / self._rate_hz, self._publish_cmd
        )

        self.get_logger().info(
            f"auto_drive_node ready — variable-speed profile over "
            f"{self._pipe_len} m (gaps at 1.2-2.0 m and 2.8-3.4 m)"
        )

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        if self._start_x is None:
            self._start_x = x
            self.get_logger().info(f"Odometry received — start x={x:.3f}")
        self._distance = abs(x - self._start_x)

    # ------------------------------------------------------------------
    def _publish_cmd(self):
        if self._done:
            self._pub.publish(Twist())   # zero velocity — stay stopped
            return

        if self._start_x is None:
            # Physics not running yet — wait silently
            return

        if self._distance >= self._pipe_len:
            self._done = True
            self._pub.publish(Twist())
            self.get_logger().info(
                f"Reached {self._distance:.2f} m — stopped."
            )
            return

        cmd = Twist()
        cmd.linear.x = _speed_at(self._distance)
        self._pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AutoDrive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
