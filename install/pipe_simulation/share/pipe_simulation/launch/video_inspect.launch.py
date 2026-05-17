"""
video_inspect.launch.py
-----------------------
Full pipe-inspection perception pipeline driven by a pre-recorded video file.

No Gazebo or robot simulation is required — just a video file and RViz.

Pipeline:
  video_ingest_node           reads video → /camera/image_raw
  cylinder_unwrap_node        unwraps fisheye → /pipe/unwrapped
  coverage_mapper_node        accumulates coverage → /pipe/coverage
                              (use_odom=False: one depth slice per frame)
  cylinder_visualizer_3d_node heatmap → 3-D cylinder marker → /visualization/cylinder
  rviz2                       displays the growing cylinder

In video mode (use_odom=False) there is no odometry.  The 3D cylinder
grows as a straight line with each frame, accumulating coverage data
from the video regardless of the physical path.

Usage:
  ros2 launch pipe_simulation video_inspect.launch.py

  # Override video path
  ros2 launch pipe_simulation video_inspect.launch.py \\
      video_path:=/home/shrujans/xolo/videos/pipe.mp4

  # Adjust playback speed and outer ring
  ros2 launch pipe_simulation video_inspect.launch.py \\
      fps:=15.0 outer_fraction:=0.40
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_sim = get_package_share_directory("pipe_simulation")

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value=os.path.expanduser("~/xolo/videos/pipe.mp4"),
        description="Absolute path to the pipe inspection video file.",
    )
    fps_arg = DeclareLaunchArgument(
        "fps",
        default_value="10.0",
        description=(
            "Playback rate in frames per second.  "
            "Lower values let the 3-D cylinder update more slowly.  Default: 10."
        ),
    )
    pipe_radius_arg = DeclareLaunchArgument(
        "pipe_radius",
        default_value="0.15",
        description="Inner pipe radius in metres.",
    )
    outer_fraction_arg = DeclareLaunchArgument(
        "outer_fraction",
        default_value="0.50",
        description=(
            "Fraction of the unwrapped frame height sampled as the pipe-wall "
            "ring for coverage extraction.  0.5 matches a 180-deg equidistant "
            "fisheye where the wall falls at 50% image radius."
        ),
    )
    slice_distance_m_arg = DeclareLaunchArgument(
        "slice_distance_m",
        default_value="0.02",
        description=(
            "Kept for parameter completeness; ignored in video mode "
            "(use_odom=False adds one slice per frame regardless)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Static world → camera TF (needed so RViz can resolve the frame)
    # ------------------------------------------------------------------ #
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_tf_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "camera"],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 1. Video ingest — publishes /camera/image_raw
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. Cylinder unwrap — /camera/image_raw → /pipe/unwrapped
    # ------------------------------------------------------------------ #
    cylinder_unwrap = Node(
        package="pipe_perception",
        executable="cylinder_unwrap_node",
        name="cylinder_unwrap_node",
        output="screen",
        parameters=[
            {"radius": LaunchConfiguration("pipe_radius")},
        ],
    )

    # ------------------------------------------------------------------ #
    # 3. Coverage mapper — frame-gated mode (use_odom=False)
    #    Adds one depth slice per incoming frame; no odometry required.
    # ------------------------------------------------------------------ #
    coverage_mapper = Node(
        package="pipe_perception",
        executable="coverage_mapper_node",
        name="coverage_mapper_node",
        output="screen",
        parameters=[
            {"outer_fraction":   LaunchConfiguration("outer_fraction")},
            {"use_odom":         False},
            {"slice_distance_m": LaunchConfiguration("slice_distance_m")},
        ],
    )

    # ------------------------------------------------------------------ #
    # 4. 3-D cylinder visualiser — waypoint-driven
    #    In video mode waypoints are all (0,0,0), so the cylinder grows
    #    as a straight stub at the origin.
    # ------------------------------------------------------------------ #
    cylinder_visualizer = Node(
        package="pipe_perception",
        executable="cylinder_visualizer_3d_node",
        name="cylinder_visualizer_3d_node",
        output="screen",
        parameters=[
            {"pipe_radius": LaunchConfiguration("pipe_radius")},
        ],
    )

    # ------------------------------------------------------------------ #
    # 5. RViz2
    # ------------------------------------------------------------------ #
    rviz_config = os.path.join(pkg_sim, "rviz", "sim.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config] if os.path.isfile(rviz_config) else [],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # LaunchDescription
    # ------------------------------------------------------------------ #
    return LaunchDescription([
        video_path_arg,
        fps_arg,
        pipe_radius_arg,
        outer_fraction_arg,
        slice_distance_m_arg,
        world_tf,
        video_ingest,
        cylinder_unwrap,
        coverage_mapper,
        cylinder_visualizer,
        rviz,
    ])
