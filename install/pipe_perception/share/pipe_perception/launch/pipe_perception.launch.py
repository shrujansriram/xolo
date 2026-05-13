"""
pipe_perception.launch.py
--------------------------
Launches the full pipe inspection perception pipeline:

    1. video_ingest_node           — reads a video file and publishes raw frames
    2. cylinder_unwrap_node        — unwraps each frame cylindrically
    3. coverage_mapper_node        — accumulates coverage and publishes heatmap
    4. cylinder_visualizer_3d_node — converts heatmap to a growing 3D cylinder

The cylinder length is computed automatically in real-time as:

    pipe_length = frames_processed * depth_per_frame

so neither pipe length nor the number of frames need to be known in advance.

Usage:
    ros2 launch pipe_perception pipe_perception.launch.py
    ros2 launch pipe_perception pipe_perception.launch.py \\
        video_path:=/abs/path/to/pipe.mp4 fps:=25.0 \\
        pipe_radius:=0.10 depth_per_frame:=0.005
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ------------------------------------------------------------------ #
    # Declare overridable launch arguments
    # ------------------------------------------------------------------ #
    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value=os.path.expanduser("~/inspectly/videos/pipe.mp4"),
        description="Absolute path to the pipe inspection video file.",
    )
    fps_arg = DeclareLaunchArgument(
        "fps",
        default_value="30.0",
        description="Playback rate in frames per second.",
    )
    pipe_radius_arg = DeclareLaunchArgument(
        "pipe_radius",
        default_value="0.15",
        description="Physical inner pipe radius in metres.",
    )
    depth_per_frame_arg = DeclareLaunchArgument(
        "depth_per_frame",
        default_value="0.01",
        description=(
            "Physical distance (metres) the camera advances per frame. "
            "Controls the rate at which the cylinder grows in length. "
            "Example: 0.01 m/frame at 30 fps = 0.3 m/s camera speed."
        ),
    )

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    # Static identity transform so RViz2 can resolve the "world" frame.
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_tf_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "camera"],
        output="screen",
    )

    video_ingest = Node(
        package="pipe_perception",
        executable="video_ingest_node",
        name="video_ingest_node",
        output="screen",
        parameters=[
            {"video_path": LaunchConfiguration("video_path")},
            {"fps":        LaunchConfiguration("fps")},
        ],
    )

    cylinder_unwrap = Node(
        package="pipe_perception",
        executable="cylinder_unwrap_node",
        name="cylinder_unwrap_node",
        output="screen",
        parameters=[
            {"radius": LaunchConfiguration("pipe_radius")},
        ],
    )

    coverage_mapper = Node(
        package="pipe_perception",
        executable="coverage_mapper_node",
        name="coverage_mapper_node",
        output="screen",
    )

    cylinder_visualizer_3d = Node(
        package="pipe_perception",
        executable="cylinder_visualizer_3d_node",
        name="cylinder_visualizer_3d_node",
        output="screen",
        parameters=[
            {"pipe_radius":      LaunchConfiguration("pipe_radius")},
            {"depth_per_frame":  LaunchConfiguration("depth_per_frame")},
        ],
    )

    # ------------------------------------------------------------------ #
    # LaunchDescription
    # ------------------------------------------------------------------ #
    return LaunchDescription([
        video_path_arg,
        fps_arg,
        pipe_radius_arg,
        depth_per_frame_arg,
        world_tf,
        video_ingest,
        cylinder_unwrap,
        coverage_mapper,
        cylinder_visualizer_3d,
    ])
