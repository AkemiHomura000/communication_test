#!/bin/bash

# 工作空间根目录（脚本所在目录的上一级）
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_FILE="$WS_DIR/install/setup.bash"
ROS_LOG_DIR="${ROS_LOG_DIR:-$WS_DIR/running_log/log}"

if [ ! -f "$SETUP_FILE" ]; then
    echo "[ERROR] 未找到工作空间 setup 文件: $SETUP_FILE"
    echo "请先执行 colcon build"
    exit 1
fi

mkdir -p "$ROS_LOG_DIR"

# 在新终端中执行命令的辅助函数
open_terminal() {
    local title="$1"
    local cmd="$2"
    local full_cmd="source $SETUP_FILE && export ROS_LOG_DIR=\"$ROS_LOG_DIR\" && $cmd; exec bash"

    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal --title="$title" -- bash -c "$full_cmd" &
    elif command -v xterm &>/dev/null; then
        xterm -T "$title" -e bash -c "$full_cmd" &
    elif command -v konsole &>/dev/null; then
        konsole --new-tab -p tabtitle="$title" -e bash -c "$full_cmd" &
    else
        echo "[ERROR] 未找到可用的终端模拟器 (gnome-terminal / xterm / konsole)"
        exit 1
    fi
}

echo "启动 sp_nav navigation stack ..."

# 启动导航主程序
open_terminal "tf2" "ros2 run tf2_ros static_transform_publisher 0 0.15 0 0 0 0 base_link livox_frame"
open_terminal "tf21" "ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map lidar_odom"
open_terminal "livox_drvier" "ros2 launch livox_ros_driver2 msg_MID360_launch.py"
sleep 1
open_terminal "point_lio" "ros2 launch point_lio mapping_mid360.launch.py"
sleep 1
open_terminal "decision" "ros2 launch sp_decision decision.launch.py"
sleep 1
open_terminal "sp_nav" "ros2 launch sp_nav_bringup sp_nav.launch.py"
sleep 1
open_terminal "controller" "ros2 launch sp_controller_server controller.launch.py "
sleep 1
open_terminal "sentry_communicator" "ros2 launch sentry_communicator sentry_communicator.launch.py"
sleep 1
open_terminal "pc_filter" "ros2 launch pc_filter pc_filter.launch.py"
sleep 1
open_terminal "sp_map_server" "ros2 launch sp_map_server local_costmap_generator.launch.py"
sleep 1
open_terminal "dynamic_obstacle_detector" "ros2 launch dynamic_obstacle_detector dynamic_obstacle_detector.launch.py"
sleep 1
# open_terminal "rosbag recorder" "ros2 launch record_manager record_manager.launch.py"
echo "所有节点已在新终端中启动。"
