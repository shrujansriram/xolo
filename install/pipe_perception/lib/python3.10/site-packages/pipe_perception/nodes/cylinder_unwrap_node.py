"""
cylinder_unwrap_node.py
-----------------------
ROS2 node that subscribes to raw camera frames, applies the cylindrical
unwrapping transform, and publishes the result for downstream nodes.

Subscriptions
-------------
    /camera/image_raw  (sensor_msgs/Image)  — BGR frames from VideoIngestNode

Publications
------------
    /pipe/unwrapped    (sensor_msgs/Image)  — cylindrical projection (H x 360)

Usage (after building the package):
    ros2 run pipe_perception cylinder_unwrap_node
    ros2 run pipe_perception cylinder_unwrap_node --ros-args -p radius:=0.10
"""

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image

from pipe_perception.algorithms.cylinder_unwrap import CylinderUnwrapper


class CylinderUnwrapNode(Node):
    """
    Subscribes to /camera/image_raw, unwraps each frame cylindrically,
    and republishes on /pipe/unwrapped.

    Parameters (ROS2)
    -----------------
    radius : float
        Physical inner pipe radius in metres, forwarded to CylinderUnwrapper.
        Default: 0.15
    """

    SUB_TOPIC = "/camera/image_raw"
    PUB_TOPIC = "/pipe/unwrapped"

    def __init__(self) -> None:
        super().__init__("cylinder_unwrap_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("radius", 0.15)
        radius: float = float(self.get_parameter("radius").value)

        # ------------------------------------------------------------------ #
        # Algorithm + bridge
        # ------------------------------------------------------------------ #
        self._unwrapper = CylinderUnwrapper(radius=radius)
        self._bridge = CvBridge()

        # ------------------------------------------------------------------ #
        # Subscriber / Publisher
        # ------------------------------------------------------------------ #
        self._subscriber = self.create_subscription(
            Image,
            self.SUB_TOPIC,
            self._image_callback,
            qos_profile=10,
        )
        self._publisher = self.create_publisher(Image, self.PUB_TOPIC, qos_profile=10)

        # ------------------------------------------------------------------ #
        # Startup log
        # ------------------------------------------------------------------ #
        self.get_logger().info(
            f"CylinderUnwrapNode ready.\n"
            f"  Subscribing : {self.SUB_TOPIC}\n"
            f"  Publishing  : {self.PUB_TOPIC}\n"
            f"  Pipe radius : {radius} m"
        )

    # ---------------------------------------------------------------------- #
    # Callback
    # ---------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        """Convert incoming Image → BGR array → unwrap → publish."""
        # ROS2 Image → OpenCV BGR
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge decode failed: {exc}")
            return

        # Cylindrical unwrap
        try:
            unwrapped = self._unwrapper.unwrap_frame(frame)
        except ValueError as exc:
            self.get_logger().warning(f"unwrap_frame rejected frame: {exc}")
            return

        # OpenCV BGR → ROS2 Image
        try:
            out_msg: Image = self._bridge.cv2_to_imgmsg(unwrapped, encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge encode failed: {exc}")
            return

        # Preserve the original timestamp and frame_id for synchronisation
        out_msg.header = msg.header

        self._publisher.publish(out_msg)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    node: CylinderUnwrapNode | None = None
    try:
        node = CylinderUnwrapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
