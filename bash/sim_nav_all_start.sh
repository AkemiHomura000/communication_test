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
open_terminal "decision" "ros2 launch sp_decision decision.launch.py"
sleep 1
open_terminal "sp_nav" "ros2 launch sp_nav_bringup sp_nav.launch.py"
sleep 1
open_terminal "controller" "ros2 launch sp_controller_server controller_minco.launch.py"
sleep 1
open_terminal "mgp_gui" "ros2 run map_robot_gui map_drag_robot --ros-args   -p map_yaml:=/home/rm/Desktop/sp_nav_26/src/tools/map_process/pgm/rmul_map.yaml   -p init_x:=1.25 -p init_y:=-1.75   -p init_x2:=18.0 -p init_y2:=5.0 -p window_width:=1600 -p window_height:=1000   -p robot_radius:=0.25   -p v_max:=3.0 -p a_max:=3.0   -p treat_unknown_as_occupied:=true
"
sleep 1
echo "所有节点已在新终端中启动。"
