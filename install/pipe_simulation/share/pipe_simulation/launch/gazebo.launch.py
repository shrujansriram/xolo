"""
gazebo.launch.py
----------------
Minimal launch: Gazebo Harmonic + robot spawn + ros_gz_bridge.

What runs
  1. gz sim        -- Gazebo Harmonic with pipe.world
  2. xacro         -- expands pipe_bot.urdf.xacro to a URDF string
  3. robot_state_publisher -- publishes /tf from the URDF
  4. spawn_entity  -- drops the robot into Gazebo at x=0.3 y=0 z=-0.115
  5. ros_gz_bridge -- bridges camera, odom, cmd_vel, imu, tf, joint_states

What does NOT run here (added in sim.launch.py, Step 12)
  - video_ingest_node / cylinder_unwrap_node / coverage_mapper_node
  - cylinder_visualizer_3d_node
  - odom_depth_node
  - rviz2

Smoke-test after build:
    ros2 launch pipe_simulation gazebo.launch.py
    ros2 topic hz /camera/image_raw          # should print ~30 Hz
    ros2 topic hz /odom                      # should print ~30 Hz
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
    pkg_share = get_package_share_directory("pipe_simulation")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo simulation clock",
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    # ------------------------------------------------------------------
    # Gazebo resource path — lets Gazebo find meshes/pipe_hollow.stl
    # by bare URI  "meshes/pipe_hollow.stl"
    # ------------------------------------------------------------------
    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=pkg_share,
    )

    # ------------------------------------------------------------------
    # Start Gazebo Harmonic with our pipe world
    # ------------------------------------------------------------------
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-r -s {os.path.join(pkg_share, 'worlds', 'pipe.world')}",
        }.items(),
    )

    # ------------------------------------------------------------------
    # Expand URDF with xacro
    # ------------------------------------------------------------------
    robot_description_content = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                os.path.join(pkg_share, "urdf", "pipe_bot.urdf.xacro"),
            ]
        ),
        value_type=str,
    )
    robot_description = {"robot_description": robot_description_content}

    # ------------------------------------------------------------------
    # robot_state_publisher: broadcasts TF from URDF joint transforms
    # ------------------------------------------------------------------
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # ------------------------------------------------------------------
    # Spawn the robot inside Gazebo at the pipe entrance
    #   x=0.3  : 0.3 m inside the pipe (pipe starts at x=0)
    #   y=0    : centred laterally
    #   z=-0.115 : body centre; wheels rest on pipe floor at z=-0.15
    # ------------------------------------------------------------------
    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_pipe_bot",
        arguments=[
            "-name",   "pipe_bot",
            "-topic",  "robot_description",
            "-x",      "0.3",
            "-y",      "0.0",
            "-z",      "-0.115",
            "-R",      "0.0",
            "-P",      "0.0",
            "-Y",      "0.0",
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # ros_gz_bridge: relay topics between Gazebo and ROS2
    # ------------------------------------------------------------------
    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[
            {
                "config_file": os.path.join(pkg_share, "config", "bridge.yaml"),
                "use_sim_time": use_sim_time,
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            set_gz_resource_path,
            gz_sim_launch,
            robot_state_publisher_node,
            spawn_robot_node,
            bridge_node,
        ]
    )