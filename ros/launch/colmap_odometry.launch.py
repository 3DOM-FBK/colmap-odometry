from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
    params_file = Path(LaunchConfiguration("params_file").perform(context)).expanduser()
    if not params_file.is_absolute():
        params_file = (
            Path(get_package_share_directory("colmap_odometry"))
            / "config"
            / params_file
        )

    if not params_file.is_file():
        raise RuntimeError(f"COLMAP odometry parameter file not found: {params_file}")

    return [
        Node(
            package="colmap_odometry",
            executable="colmap_odometry_node",
            name="colmap_odometry_node",
            output="screen",
            parameters=[str(params_file)],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value="default.yaml",
                description=(
                    "Parameter filename in the installed colmap_odometry/config "
                    "directory, or an absolute path."
                ),
            ),
            OpaqueFunction(function=_launch_node),
        ]
    )
