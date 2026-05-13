"""
video_ingest_node.py
--------------------
ROS2 node that reads a video file and publishes each frame as a
sensor_msgs/Image message on /camera/image_raw.

The video loops automatically when it reaches the last frame.

Usage (after building the package):
    ros2 run pipe_perception video_ingest_node
    ros2 run pipe_perception video_ingest_node --ros-args \
        -p video_path:=/abs/path/to/pipe.mp4 -p fps:=25.0
"""

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoIngestNode(Node):
    """
    Reads a video file frame-by-frame and publishes BGR images.

    Parameters (ROS2)
    -----------------
    video_path : str
        Absolute or ~-prefixed path to the video file.
        Default: ~/inspectly/videos/pipe.mp4
    fps : float
        Playback rate in frames per second.  Default: 30.0
    """

    TOPIC = "/camera/image_raw"
    LOG_INTERVAL = 30  # log a status line every N frames

    def __init__(self) -> None:
        super().__init__("video_ingest_node")

        # ------------------------------------------------------------------ #
        # ROS2 parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter(
            "video_path",
            os.path.expanduser("~/inspectly/videos/pipe.mp4"),
        )
        self.declare_parameter("fps", 30.0)

        video_path: str = os.path.expanduser(
            str(self.get_parameter("video_path").value)
        )
        fps: float = float(self.get_parameter("fps").value)

        if fps <= 0.0:
            self.get_logger().error(
                f"fps must be positive, got {fps}. Falling back to 30.0."
            )
            fps = 30.0

        # ------------------------------------------------------------------ #
        # Validate video file
        # ------------------------------------------------------------------ #
        if not os.path.isfile(video_path):
            self.get_logger().fatal(
                f"Video file not found: '{video_path}'. "
                "Place pipe.mp4 in ~/inspectly/videos/ or set the "
                "'video_path' parameter."
            )
            raise FileNotFoundError(f"Video file not found: '{video_path}'")

        # ------------------------------------------------------------------ #
        # OpenCV capture
        # ------------------------------------------------------------------ #
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            self.get_logger().fatal(
                f"OpenCV could not open video: '{video_path}'"
            )
            raise RuntimeError(f"Cannot open video: '{video_path}'")

        self.get_logger().info(
            f"Opened video '{video_path}'  "
            f"({int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
            f"publishing at {fps} fps)"
        )

        # ------------------------------------------------------------------ #
        # Publisher + cv_bridge
        # ------------------------------------------------------------------ #
        self._publisher = self.create_publisher(Image, self.TOPIC, qos_profile=10)
        self._bridge = CvBridge()
        self._frame_count: int = 0

        # ------------------------------------------------------------------ #
        # Timer-driven publish loop
        # ------------------------------------------------------------------ #
        period_s: float = 1.0 / fps
        self._timer = self.create_timer(period_s, self._timer_callback)

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #

    def _timer_callback(self) -> None:
        """Read one frame and publish; loop the video when it ends."""
        ret: bool
        frame = None
        ret, frame = self._cap.read()

        if not ret or frame is None:
            # End of file — loop back to the first frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()

            if not ret or frame is None:
                self.get_logger().error(
                    "Failed to read frame even after rewinding. "
                    "The video file may be corrupt."
                )
                return

            self.get_logger().info("Video looped back to frame 0.")

        msg: Image = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        self._publisher.publish(msg)

        self._frame_count += 1
        if self._frame_count % self.LOG_INTERVAL == 0:
            self.get_logger().info(
                f"Published frame {self._frame_count} on {self.TOPIC}"
            )

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    def destroy_node(self) -> None:
        """Release the video capture on shutdown."""
        if self._cap.isOpened():
            self._cap.release()
        super().destroy_node()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    node: VideoIngestNode | None = None
    try:
        node = VideoIngestNode()
        rclpy.spin(node)
    except FileNotFoundError:
        # Already logged as FATAL inside __init__; exit cleanly.
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
