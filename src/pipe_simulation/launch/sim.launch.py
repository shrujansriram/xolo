"""
sim.launch.py
-------------
Full pipe-inspection simulation launch.

Robot drive path (L-shaped):
  1. Travel leg1_m metres straight along +X
  2. Pause pause_s seconds
  3. Turn 90° CCW in place
  4. Travel leg2_m metres straight along +Y

The pipe mesh matches this path.  Coverage and 3D visualisation are
now driven entirely by odometry: one coverage depth-slice is added per
slice_distance_m of travel, and the 3D cylinder follows the robot's
actual poses published on /pipe/waypoints.

Parameters
  pipe_radius    — inner pipe radius in metres         (default 0.15)
  leg1_m         — first straight leg length [m]       (default 2.0)
  leg2_m         — second straight leg length [m]      (default 2.0)
  corner_radius  — corner arc radius [m]               (default 0.30)
  use_sim_time   — use Gazebo clock                    (default true)

Usage
  ros2 launch pipe_simulation sim.launch.py
  ros2 launch pipe_simulation sim.launch.py leg1_m:=3.0 leg2_m:=3.0
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
# OpaqueFunction: resolves params, generates mesh, builds all nodes
# --------------------------------------------------------------------------- #

def launch_setup(context, *args, **kwargs):
    # ------------------------------------------------------------------ #
    # Resolve launch-configuration values to Python scalars
    # ------------------------------------------------------------------ #
    pipe_radius   = float(LaunchConfiguration("pipe_radius").perform(context))
    leg1_m        = float(LaunchConfiguration("leg1_m").perform(context))
    leg2_m        = float(LaunchConfiguration("leg2_m").perform(context))
    corner_radius = float(LaunchConfiguration("corner_radius").perform(context))

    use_sim_time_str = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time     = use_sim_time_str.lower() in ("true", "1", "yes")

    # ------------------------------------------------------------------ #
    # 0. Generate the L-bend pipe mesh
    # ------------------------------------------------------------------ #
    pkg_sim  = get_package_share_directory("pipe_simulation")
    pkg_perc = get_package_share_directory("pipe_perception")
    pkg_rgs  = get_package_share_directory("ros_gz_sim")

    mesh_out = os.path.join(pkg_sim, "meshes", "pipe_hollow.stl")
    script   = os.path.join(pkg_sim, "scripts", "generate_curved_pipe_mesh.py")

    subprocess.run(
        [
            sys.executable, script,
            "--shape",         "l_bend",
            "--inner_radius",  str(pipe_radius),
            "--leg1",          str(leg1_m),
            "--leg2",          str(leg2_m),
            "--corner_radius", str(corner_radius),
            "--output",        mesh_out,
        ],
        check=True,
    )

    # ------------------------------------------------------------------ #
    # 1. Gazebo Harmonic (server-only, starts unpaused)
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
    # 3. Spawn robot at the pipe entrance
    # ------------------------------------------------------------------ #
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_pipe_bot",
        arguments=[
            "-name", "pipe_bot",
            "-topic", "robot_description",
            "-x", "0.3",
            "-y", "0.0",
            "-z", "-0.115",
            "-R", "0.0",
            "-P", "0.0",
            "-Y", "0.0",
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
    # 6. coverage_mapper_node — odom-gated mode
    #    outer_fraction=0.50: pipe wall falls at 50% image radius for the
    #    180-deg equidistant fisheye.
    #    use_odom=True: add one depth slice per slice_distance_m of travel.
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
    # 7. cylinder_visualizer_3d_node — waypoint-driven (no arc params)
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
    # 8. auto_drive_node — L-path state machine
    # ------------------------------------------------------------------ #
    auto_drive = Node(
        package="pipe_simulation",
        executable="auto_drive_node",
        name="auto_drive_node",
        parameters=[
            {"leg1_m":       leg1_m},
            {"pause_s":      2.0},
            {"turn_deg":     90.0},
            {"leg2_m":       leg2_m},
            {"drive_speed":  0.10},
            {"turn_speed":   0.40},
            {"cmd_rate_hz":  10.0},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # 9. synthetic_camera_node
    # ------------------------------------------------------------------ #
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
    # World TF: static transform world → odom (identity)
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
        description="Inner pipe radius in metres (must match STL).",
    )
    leg1_m_arg = DeclareLaunchArgument(
        "leg1_m",
        default_value="2.0",
        description="First straight section length [m].",
    )
    leg2_m_arg = DeclareLaunchArgument(
        "leg2_m",
        default_value="2.0",
        description="Second straight section length [m].",
    )
    corner_radius_arg = DeclareLaunchArgument(
        "corner_radius",
        default_value="0.30",
        description="Corner arc radius [m].",
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=pkg_sim,
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            pipe_radius_arg,
            leg1_m_arg,
            leg2_m_arg,
            corner_radius_arg,
            set_gz_resource_path,
            OpaqueFunction(function=launch_setup),
        ]
    )
