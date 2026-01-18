from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sentry_communicator',
            executable='sentry_communicator',
            name='can_bus_node',
            output='screen',
            parameters=[{'debug':False}]
        ),
    ])