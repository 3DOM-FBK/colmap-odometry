"""COLMAP for the IDEAL rectified ZED stereo pair."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "publish_tf", default_value="false",
            description="Enable for standalone VO only; Stelvio owns TF in fusion runs."),
        DeclareLaunchArgument(
            "params_file", default_value=PathJoinSubstitution([
                FindPackageShare("colmap_odometry"), "config", "ideal.yaml"])),
        Node(
            package="colmap_odometry", executable="colmap_odometry_node",
            name="colmap_odometry_node", output="screen",
            parameters=[LaunchConfiguration("params_file"), {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool),
                "publish_tf": ParameterValue(
                    LaunchConfiguration("publish_tf"), value_type=bool),
            }],
        ),
    ])
