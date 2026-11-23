"""Launch COLMAP-SLAM for the GrandTour Alphasense stereo pair."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("colmap_odometry"), "config", "grandtour.yaml"]),
            description="COLMAP odometry parameter file."),
        Node(
            package="colmap_odometry",
            executable="colmap_odometry_node",
            name="colmap_odometry_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file"), {
                "use_sim_time": LaunchConfiguration("use_sim_time")
            }],
        ),
    ])
