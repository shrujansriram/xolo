"""
auto_drive_node.py
------------------
Drives the pipe_bot forward through the pipe at a constant speed.

Behaviour
  1. Wait for /odom to confirm the robot is alive and physics is running.
  2. Drive forward at `linear_speed` m/s until `pipe_length` metres have
     been covered (measured from odometry).
  3. Stop and hold position.

Parameters (all ROS2 parameters, settable from launch or command line)
  linear_speed   (default 0.10)  — forward speed in m/s
  pipe_length    (default 4.50)  — metres to travel before stopping
  cmd_rate_hz    (default 10.0)  — rate at which cmd_vel is published
  use_sim_time   (default true)  — must match the rest of the stack
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class AutoDrive(Node):
    def __init__(self):
        super().__init__("auto_drive_node")

        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("pipe_length",  4.50)
        self.declare_parameter("cmd_rate_hz",  10.0)

        self._speed      = self.get_parameter("linear_speed").value
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
            f"auto_drive_node ready — driving {self._pipe_len} m "
            f"at {self._speed} m/s"
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
        cmd.linear.x = self._speed
        self._pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AutoDrive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
