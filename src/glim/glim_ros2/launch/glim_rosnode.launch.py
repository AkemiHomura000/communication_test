from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='/home/rm/Desktop/communication_test/src/glim/glim/config',
        description='Path to glim config directory'
    )

    dump_path_arg = DeclareLaunchArgument(
        'dump_path',
        default_value='/home/rm/Desktop/communication_test/src/glim/glim_ros2/dump',
        description='Path to save the global map'
    )

    dump_on_unload_arg = DeclareLaunchArgument(
        'dump_on_unload',
        default_value='true',
        description='Save map automatically when node is shut down'
    )

    glim_node = Node(
        package='glim_ros',
        executable='glim_rosnode',
        name='glim_rosnode',
        output='screen',
        parameters=[{
            'config_path': LaunchConfiguration('config_path'),
            'dump_path': LaunchConfiguration('dump_path'),
            'dump_on_unload': LaunchConfiguration('dump_on_unload'),
        }],
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2 with glim config'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', '/home/rm/Desktop/communication_test/src/glim/glim_ros2/rviz/glim_ros.rviz'],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        config_path_arg,
        dump_path_arg,
        dump_on_unload_arg,
        rviz_arg,
        glim_node,
        rviz_node,
    ])
