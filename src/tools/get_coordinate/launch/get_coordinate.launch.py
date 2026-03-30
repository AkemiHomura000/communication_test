from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    get_coordinate_config = os.path.join(
        get_package_share_directory('get_coordinate'),
        'config',
        'config.yaml'
    )

    return LaunchDescription([
        Node(
            package='get_coordinate',
            executable='get_coordinate',
            name='get_coordinate_1',
            output='screen',
            parameters=[{'config_file': get_coordinate_config}]
        ),
    ])