import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("global_relocalization")

    # ── Launch 参数 ────────────────────────────────────────────────────────
    map_pcd_path_arg = DeclareLaunchArgument(
        "map_pcd_path",
        default_value="",
        description="地图 PCD 文件的绝对路径（map 坐标系）。",
    )

    cloud_topic_arg = DeclareLaunchArgument(
        "cloud_sub_topic",
        default_value="/cloud_registered",
        description="实时点云话题（lidar_odom 坐标系，通常由 point_lio 发布）。",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "params.yaml"]),
        description="YAML 参数文件路径。",
    )

    # ── 节点 ───────────────────────────────────────────────────────────────
    relocalization_node = Node(
        package="global_relocalization",
        executable="global_relocalization_node",
        name="global_relocalization_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            # 命令行参数优先级高于 yaml（如需覆盖可取消注释）
            # {
            #     "map_pcd_path": LaunchConfiguration("map_pcd_path"),
            #     "cloud_sub_topic": LaunchConfiguration("cloud_sub_topic"),
            # },
        ],
    )

    return LaunchDescription(
        [
            # map_pcd_path_arg,
            # cloud_topic_arg,
            params_file_arg,
            relocalization_node,
        ]
    )
