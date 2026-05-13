"""
odom_depth_node.py
------------------
Derives depth_per_frame from robot wheel odometry and forwards it live
to cylinder_visualizer_3d_node via the ROS2 SetParameters service.

How it works
    The fisheye camera is fixed to the robot body.  As the robot moves
    forward at speed v (m/s), the camera advances v metres per second.
    At camera_fps frames per second the per-frame depth advance is:

        depth_per_frame = v / camera_fps   (metres/frame)

    1. /odom callback  → compute instantaneous speed, apply EMA smoothing.
    2. Timer (update_period)  → if value changed, call
         /cylinder_visualizer_3d_node/set_parameters

Parameters (ROS2)
-----------------
camera_fps    : float  — camera frame rate in Hz        (default 30.0)
update_period : float  — SetParameters call interval s  (default 1.0)
alpha         : float  — EMA smoothing 0 < α ≤ 1        (default 0.3)
target_node   : str    — visualiser node name
                         (default "cylinder_visualizer_3d_node")
"""

import math

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter as RosParameter
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node


class OdomDepthNode(Node):
    """Convert odometry speed → depth_per_frame, push via SetParameters."""

    def __init__(self) -> None:
        super().__init__("odom_depth_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("camera_fps",    30.0)
        self.declare_parameter("update_period",  1.0)
        self.declare_parameter("alpha",          0.3)
        self.declare_parameter("target_node",    "cylinder_visualizer_3d_node")

        self._camera_fps   = float(self.get_parameter("camera_fps").value)
        self._alpha        = float(self.get_parameter("alpha").value)
        target_node        = str(self.get_parameter("target_node").value)
        update_period      = float(self.get_parameter("update_period").value)

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._prev_x: float | None     = None
        self._prev_y: float | None     = None
        self._prev_stamp: float | None = None
        self._smoothed_speed: float    = 0.0   # m/s, EMA-filtered
        self._last_sent_depth: float   = -1.0  # sentinel: never sent yet

        # ------------------------------------------------------------------ #
        # SetParameters service client for the visualiser
        # ------------------------------------------------------------------ #
        svc_name = f"/{target_node}/set_parameters"
        self._set_param_client = self.create_client(SetParameters, svc_name)

        # ------------------------------------------------------------------ #
        # Odometry subscriber
        # ------------------------------------------------------------------ #
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)

        # ------------------------------------------------------------------ #
        # Timer — push depth_per_frame at update_period intervals
        # ------------------------------------------------------------------ #
        self.create_timer(update_period, self._push_depth_per_frame)

        self.get_logger().info(
            f"OdomDepthNode ready  |  camera_fps={self._camera_fps} Hz  "
            f"alpha={self._alpha}  target={svc_name}"
        )

    # ---------------------------------------------------------------------- #
    # Odometry callback — maintain EMA-smoothed speed estimate
    # ---------------------------------------------------------------------- #

    def _odom_callback(self, msg: Odometry) -> None:
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._prev_x is None:
            self._prev_x, self._prev_y, self._prev_stamp = x, y, stamp
            return

        dt = stamp - self._prev_stamp
        if dt <= 1e-6:
            return

        dist  = math.hypot(x - self._prev_x, y - self._prev_y)
        speed = dist / dt

        # Exponential moving average to smooth wheel-slip noise
        self._smoothed_speed = (
            self._alpha * speed + (1.0 - self._alpha) * self._smoothed_speed
        )

        self._prev_x, self._prev_y, self._prev_stamp = x, y, stamp

    # ---------------------------------------------------------------------- #
    # Timer callback — push updated depth_per_frame if it changed
    # ---------------------------------------------------------------------- #

    def _push_depth_per_frame(self) -> None:
        if not self._set_param_client.service_is_ready():
            # Visualiser not running yet; will retry next tick
            return

        depth = self._smoothed_speed / self._camera_fps

        # Skip if value has not changed by more than 5% (reduces chatter)
        if self._last_sent_depth >= 0.0:
            ref = max(depth, self._last_sent_depth, 1e-9)
            if abs(depth - self._last_sent_depth) / ref < 0.05:
                return

        param_val = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=depth,
        )
        ros_param = RosParameter(name="depth_per_frame", value=param_val)
        request   = SetParameters.Request(parameters=[ros_param])

        future = self._set_param_client.call_async(request)
        future.add_done_callback(self._on_set_param_done)

        self._last_sent_depth = depth
        self.get_logger().debug(
            f"depth_per_frame → {depth:.5f} m  "
            f"(speed {self._smoothed_speed:.3f} m/s)"
        )

    def _on_set_param_done(self, future) -> None:
        try:
            result = future.result()
            if result and not result.results[0].successful:
                self.get_logger().warning(
                    f"SetParameters rejected: {result.results[0].reason}"
                )
        except Exception as exc:
            self.get_logger().error(f"SetParameters call failed: {exc}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    node: OdomDepthNode | None = None
    try:
        node = OdomDepthNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()