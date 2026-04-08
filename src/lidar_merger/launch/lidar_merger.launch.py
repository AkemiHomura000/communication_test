from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lidar_merger'),
        'config',
        'extrinsic.yaml'
    )

    return LaunchDescription([
        Node(
            package='lidar_merger',
            executable='lidar_merger_node',
            name='lidar_merger',
            output='screen',
            parameters=[config],
        ),
    ])
