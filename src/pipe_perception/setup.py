from setuptools import find_packages, setup

package_name = "pipe_perception"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Install package.xml so ROS2 can find the package
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # Install launch files
        (f"share/{package_name}/launch", ["pipe_perception/launch/pipe_perception.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainer",
    maintainer_email="maintainer@example.com",
    description="ROS2 pipe inspection perception pipeline.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "video_ingest_node = pipe_perception.nodes.video_ingest_node:main",
            "cylinder_unwrap_node = pipe_perception.nodes.cylinder_unwrap_node:main",
            "coverage_mapper_node = pipe_perception.nodes.coverage_mapper_node:main",
            "cylinder_visualizer_3d_node = pipe_perception.nodes.cylinder_visualizer_3d_node:main",
        ],
    },
)
