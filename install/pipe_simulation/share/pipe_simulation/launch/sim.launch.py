"""
sim.launch.py
-------------
Full pipe-inspection simulation — straight pipe with four half-disc wall
obstacles at fixed positions along the pipe.

Pipe: straight along +X for pipe_length metres.

Obstacles (hardcoded per specification):
  x = 0.20 m : left  half of cross-section blocked  (world +Y side)
  x = 1.20 m : left  half of cross-section blocked
  x = 1.90 m : right half of cross-section blocked  (world -Y side)
  x = 2.60 m : right half of cross-section blocked

Robot drives straight from x≈0.05 to x≈pipe_length−0.10, then stops.

Parameters
  pipe_radius    — inner pipe radius in metres         (default 0.15)
  pipe_length    — total straight pipe length [m]      (default 3.0)
  use_sim_time   — use Gazebo clock                    (default true)

Usage
  ros2 launch pipe_simulation sim.launch.py
  ros2 launch pipe_simulation sim.launch.py pipe_length:=3.5
  ros2 launch pipe_simulation sim.launch.py pipe_radius:=0.12
"""

import os
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# --------------------------------------------------------------------------- #
# Sphere obstacle — centred in the pipe cross-section at x=1.5 m, r=0.04 m.
# Robot dodges LEFT (y > 0); right side of pipe wall near x=1.5 is missed.
# _WALLS drives auto_drive_node to swerve left at x=1.5 (same sign as "right").
# _SPHERE is forwarded to synthetic_camera_node for ray-sphere intersection.
# --------------------------------------------------------------------------- #
_WALLS  = "1.5:right"      # auto_drive: dodge left at x=1.5 m
_SPHERE = "1.5:0.04"       # camera: sphere at x=1.5, radius 0.04 m


# --------------------------------------------------------------------------- #
# OpaqueFunction: resolves params, generates mesh, builds all nodes
# --------------------------------------------------------------------------- #

def launch_setup(context, *args, **kwargs):
    # ------------------------------------------------------------------ #
    # Resolve launch-configuration values to Python scalars
    # ------------------------------------------------------------------ #
    pipe_radius   = float(LaunchConfiguration("pipe_radius").perform(context))
    pipe_length   = float(LaunchConfiguration("pipe_length").perform(context))

    use_sim_time_str = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time     = use_sim_time_str.lower() in ("true", "1", "yes")

    # ------------------------------------------------------------------ #
    # 0. Generate the straight pipe mesh
    # ------------------------------------------------------------------ #
    pkg_sim  = get_package_share_directory("pipe_simulation")
    pkg_perc = get_package_share_directory("pipe_perception")
    pkg_rgs  = get_package_share_directory("ros_gz_sim")

    mesh_out = os.path.join(pkg_sim, "meshes", "pipe_hollow.stl")
    script   = os.path.join(pkg_sim, "scripts", "generate_curved_pipe_mesh.py")

    subprocess.run(
        [
            sys.executable, script,
            "--shape",          "arc",
            "--bend_angle_deg", "0",
            "--pipe_length",    str(pipe_length),
            "--inner_radius",   str(pipe_radius),
            "--output",         mesh_out,
        ],
        check=True,
    )

    # ------------------------------------------------------------------ #
    # 1. Gazebo Harmonic (server-only, headless)
    # ------------------------------------------------------------------ #
    world_path = os.path.join(pkg_sim, "worlds", "pipe.world")
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rgs, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r -s {world_path}"}.items(),
    )

    # ------------------------------------------------------------------ #
    # 2. robot_state_publisher
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 3. Spawn robot at the pipe entrance (before first wall at 0.2 m)
    # ------------------------------------------------------------------ #
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_pipe_bot",
        arguments=[
            "-name",  "pipe_bot",
            "-topic", "robot_description",
            "-x",     "0.05",
            "-y",     "0.0",
            "-z",     "-0.115",
            "-R",     "0.0",
            "-P",     "0.0",
            "-Y",     "0.0",
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 4. ros_gz_bridge
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 5. cylinder_unwrap_node
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 6. coverage_mapper_node — odom-gated (one depth slice per 0.02 m)
    # ------------------------------------------------------------------ #
    coverage_mapper = Node(
        package="pipe_perception",
        executable="coverage_mapper_node",
        name="coverage_mapper_node",
        parameters=[
            {"outer_fraction":  0.50},
            {"use_odom":        True},
            {"slice_distance_m": 0.02},
            {"use_sim_time":    use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 7. cylinder_visualizer_3d_node — waypoint-driven
    # ------------------------------------------------------------------ #
    cylinder_visualizer = Node(
        package="pipe_perception",
        executable="cylinder_visualizer_3d_node",
        name="cylinder_visualizer_3d_node",
        parameters=[
            {"pipe_radius":  pipe_radius},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 8. auto_drive_node — straight drive with proportional Y-tracking
    #    wall avoidance.  The robot swings laterally to the clear half
    #    of the cross-section before each wall, passes through, then
    #    returns to the centreline.
    # ------------------------------------------------------------------ #
    auto_drive = Node(
        package="pipe_simulation",
        executable="auto_drive_node",
        name="auto_drive_node",
        parameters=[
            {"pipe_length":  pipe_length},
            {"drive_speed":  0.10},
            {"y_offset":     0.08},   # clear sphere at centre: robot at y=+0.08 m
            {"avoid_range":  0.30},
            {"kp_y":         6.0},
            {"omega_max":    0.40},
            {"cmd_rate_hz":  10.0},
            {"use_sim_time": use_sim_time},
            {"walls":        _WALLS},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 9. synthetic_camera_node — with sphere obstacle
    # ------------------------------------------------------------------ #
    synthetic_camera = Node(
        package="pipe_simulation",
        executable="synthetic_camera_node",
        name="synthetic_camera_node",
        parameters=[
            {"pipe_radius":       pipe_radius},
            {"image_size":        320},
            {"update_rate_hz":    10.0},
            {"use_sim_time":      use_sim_time},
            {"sphere_obstacles":  _SPHERE},  # "x:radius" centred in cross-section
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 10. RViz2
    # ------------------------------------------------------------------ #
    rviz_config = os.path.join(pkg_sim, "rviz", "sim.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config] if os.path.isfile(rviz_config) else [],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # World TF: world → odom (identity)
    # ------------------------------------------------------------------ #
    world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_odom_tf",
        arguments=["0", "0", "0", "0", "0", "0", "world", "odom"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return [
        gz_sim_launch,
        robot_state_publisher,
        spawn_robot,
        bridge,
        cylinder_unwrap,
        coverage_mapper,
        cylinder_visualizer,
        auto_drive,
        synthetic_camera,
        world_tf,
        rviz,
    ]


# --------------------------------------------------------------------------- #
# generate_launch_description
# --------------------------------------------------------------------------- #

def generate_launch_description():
    pkg_sim = get_package_share_directory("pipe_simulation")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo simulation clock for all nodes.",
    )
    pipe_radius_arg = DeclareLaunchArgument(
        "pipe_radius",
        default_value="0.15",
        description="Inner pipe radius in metres.",
    )
    pipe_length_arg = DeclareLaunchArgument(
        "pipe_length",
        default_value="3.0",
        description="Total straight pipe length in metres.",
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=pkg_sim,
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            pipe_radius_arg,
            pipe_length_arg,
            set_gz_resource_path,
            OpaqueFunction(function=launch_setup),
        ]
    )
