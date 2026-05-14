"""
sim.launch.py
-------------
Full pipe-inspection simulation launch.

Nodes started
  Gazebo layer
    1. gz sim              — Gazebo Harmonic with pipe.world
    2. robot_state_pub     — TF from URDF joint states
    3. spawn pipe_bot      — places robot inside the pipe
    4. ros_gz_bridge       — bridges all topics between Gazebo and ROS2

  Perception pipeline  (pipe_perception package)
    5. cylinder_unwrap_node    — fisheye → cylindrical unwrap
    6. coverage_mapper_node    — accumulates coverage heatmap
                                 outer_fraction=0.50  (fisheye 180-deg:
                                 pipe wall at 90-deg → 50% image radius)
    7. cylinder_visualizer_3d  — heatmap → 3D RViz2 marker

  Simulation helpers  (pipe_simulation package)
    8. odom_depth_node         — /odom speed → depth_per_frame (live)

  Visualisation
    9. rviz2                   — pre-configured RViz2 session

Note: video_ingest_node is NOT launched — Gazebo provides /camera/image_raw
directly, which cylinder_unwrap_node already subscribes to.

Key parameters
  pipe_radius     (default 0.15) — must match the STL inner radius
  outer_fraction  (default 0.50) — tuned for 180-deg equidistant fisheye
  use_sim_time    (default true)  — all nodes use Gazebo clock

Usage
  ros2 launch pipe_simulation sim.launch.py
  ros2 launch pipe_simulation sim.launch.py pipe_radius:=0.15
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_sim   = get_package_share_directory("pipe_simulation")
    pkg_perc  = get_package_share_directory("pipe_perception")
    pkg_rgs   = get_package_share_directory("ros_gz_sim")

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo simulation clock for all nodes.",
    )
    pipe_radius_arg = DeclareLaunchArgument(
        "pipe_radius",
        default_value="0.15",
        description="Inner pipe radius in metres (must match STL).",
    )

    use_sim_time  = LaunchConfiguration("use_sim_time")
    pipe_radius   = LaunchConfiguration("pipe_radius")

    # ------------------------------------------------------------------
    # Expose package share to Gazebo so it finds meshes/pipe_hollow.stl
    # ------------------------------------------------------------------
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=pkg_sim,
    )

    # ------------------------------------------------------------------
    # 1. Gazebo Harmonic
    # ------------------------------------------------------------------
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rgs, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-r -s {os.path.join(pkg_sim, 'worlds', 'pipe.world')}",
        }.items(),
    )

    # ------------------------------------------------------------------
    # 2. robot_state_publisher
    # ------------------------------------------------------------------
    robot_description_content = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                os.path.join(pkg_sim, "urdf", "pipe_bot.urdf.xacro"),
            ]
        ),
        value_type=str,
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description_content},
            {"use_sim_time": use_sim_time},
        ],
    )

    # ------------------------------------------------------------------
    # 3. Spawn robot inside the pipe
    #    x=0.3 m inside pipe (pipe starts at x=0 after pitch rotation)
    #    z=-0.115 m so wheels rest on pipe floor (inner floor at z=-0.15)
    # ------------------------------------------------------------------
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_pipe_bot",
        arguments=[
            "-name", "pipe_bot",
            "-topic", "robot_description",
            "-x",    "0.3",
            "-y",    "0.0",
            "-z",    "-0.115",
            "-R",    "0.0",
            "-P",    "0.0",
            "-Y",    "0.0",
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 4. ros_gz_bridge — relay all sensor + control topics
    # ------------------------------------------------------------------
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[
            {
                "config_file": os.path.join(pkg_sim, "config", "bridge.yaml"),
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 5. cylinder_unwrap_node
    #    Subscribes to /camera/image_raw (same topic Gazebo publishes).
    #    No video_ingest_node needed in simulation.
    # ------------------------------------------------------------------
    cylinder_unwrap = Node(
        package="pipe_perception",
        executable="cylinder_unwrap_node",
        name="cylinder_unwrap_node",
        parameters=[
            {"radius": pipe_radius},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 6. coverage_mapper_node
    #    outer_fraction=0.50: with a 180-deg equidistant fisheye the pipe
    #    wall (at 90-deg from the optical axis) falls at exactly 50% of
    #    the image radius in the unwrapped frame.
    # ------------------------------------------------------------------
    coverage_mapper = Node(
        package="pipe_perception",
        executable="coverage_mapper_node",
        name="coverage_mapper_node",
        parameters=[
            {"outer_fraction": 0.50},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 7. cylinder_visualizer_3d_node
    #    depth_per_frame starts at the default (0.01 m); odom_depth_node
    #    will update it in real-time via SetParameters.
    # ------------------------------------------------------------------
    cylinder_visualizer = Node(
        package="pipe_perception",
        executable="cylinder_visualizer_3d_node",
        name="cylinder_visualizer_3d_node",
        parameters=[
            {"pipe_radius":     pipe_radius},
            {"depth_per_frame": 0.01},
            {"use_sim_time":    use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 8. odom_depth_node
    #    Watches /odom and keeps depth_per_frame in sync with robot speed.
    # ------------------------------------------------------------------
    odom_depth = Node(
        package="pipe_simulation",
        executable="odom_depth_node",
        name="odom_depth_node",
        parameters=[
            {"camera_fps":    10.0},
            {"update_period":  1.0},
            {"alpha":          0.3},
            {"use_sim_time":  use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 9. auto_drive_node
    #    Drives the robot forward 4.5 m through the pipe automatically.
    #    Waits for /odom before moving, so it is safe to launch early.
    # ------------------------------------------------------------------
    auto_drive = Node(
        package="pipe_simulation",
        executable="auto_drive_node",
        name="auto_drive_node",
        parameters=[
            {"linear_speed":  0.10},
            {"pipe_length":   4.50},
            {"cmd_rate_hz":   10.0},
            {"use_sim_time":  use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 10. synthetic_camera_node
    #     Publishes /camera/image_raw without Ogre2/Gazebo.
    #     Bypasses the EGL_MESA_device_software fixed-function crash on WSL2.
    # ------------------------------------------------------------------
    synthetic_camera = Node(
        package="pipe_simulation",
        executable="synthetic_camera_node",
        name="synthetic_camera_node",
        parameters=[
            {"pipe_radius":    pipe_radius},
            {"image_size":     320},
            {"update_rate_hz": 10.0},
            {"use_sim_time":   use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 11. RViz2
    # ------------------------------------------------------------------
    rviz_config = os.path.join(pkg_sim, "rviz", "sim.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config] if os.path.isfile(rviz_config) else [],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ------------------------------------------------------------------
    # World TF: static transform world → odom (identity)
    # ------------------------------------------------------------------
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_odom_tf",
        arguments=["0", "0", "0", "0", "0", "0", "world", "odom"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            pipe_radius_arg,
            set_gz_resource_path,
            # Gazebo
            gz_sim_launch,
            robot_state_publisher,
            spawn_robot,
            bridge,
            # Perception pipeline
            cylinder_unwrap,
            coverage_mapper,
            cylinder_visualizer,
            odom_depth,
            auto_drive,
            synthetic_camera,
            # Visualisation
            world_tf,
            rviz,
        ]
    )