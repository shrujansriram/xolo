from setuptools import find_packages, setup
import os
from glob import glob

package_name = "pipe_simulation"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # Launch files
        (f"share/{package_name}/launch", glob("launch/*.py")),
        # URDF / xacro
        (f"share/{package_name}/urdf", glob("urdf/*")),
        # Gazebo worlds
        (f"share/{package_name}/worlds", glob("worlds/*")),
        # Meshes
        (f"share/{package_name}/meshes", glob("meshes/*")),
        # Config (bridge YAML, etc.)
        (f"share/{package_name}/config", glob("config/*")),
        # RViz configs
        (f"share/{package_name}/rviz", glob("rviz/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Gazebo Harmonic pipe inspection simulation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "odom_depth_node = pipe_simulation.odom_depth_node:main",
            "auto_drive_node = pipe_simulation.auto_drive_node:main",
            "synthetic_camera_node = pipe_simulation.synthetic_camera_node:main",
        ],
    },
)
