# lidar_extrinsic_calibration 工具包说明

本目录下包含一组用于 **雷达外参标定** 及相关辅助标定的 ROS 2 Python 工具节点。这些工具均面向搭载云台（gimbal）的差速/麦轮底盘机器人，配合 LIO（激光惯性里程计，如 Point-LIO）使用。

---

## 工具一览

| 文件名 | 节点名 / 类型 | 功能简述 |
|--------|--------------|---------|
| `estimate_delay_by_gyro_corr.py` | `GyroDelayEstimator` | 通过角速度归一化互相关估计 LIO 与底盘的时间戳延迟 |
| `extrinsic_test.py` | `ChassisToMappedOdom` | 在线将底盘速度通过外参转换为 LIO 坐标系下的里程计并发布 |
| `offline_extrinsic_calib_se2.py` | 离线脚本 | 离线从 rosbag 读取数据，用 SE(2) 最小二乘优化标定雷达-底盘外参 |
| `P_gl_x_y.py` | `OdomOffsetEstimator` | 利用 LIO 里程计的速度与角速度估计雷达相对底盘旋转中心的偏移量（机体系 XY） |
| `P_gl_xy.py` | `LioPathRadiusFitter` | 订阅 LIO 发布的 Path，用 Kåsa 圆拟合算法拟合运动轨迹半径，辅助外参估计 |
| `imu_calibration.py` | `ImuLevelAndRepublishNode` | 对 IMU 原始数据做水平校正（估计静态 roll/pitch 并旋转加速度和角速度），重发布校正后的 IMU 话题 |
| `integrate_pointcloud_to_pcd.py` | `IntegratePointCloudToPCD` | 订阅多个 PointCloud2 话题，累积若干帧后保存为 `.pcd` 文件（支持 ASCII/Binary，支持 intensity） |
| `timestamp_test.py` | `TopicTimePrinter` | 诊断工具：打印 `/Odometry` 和 `/chassis_info` 每条消息的接收时间、消息时间戳及二者之差，用于判断时间延迟 |
| `calib_lidar_yaw.py` | `LidarYawCalibNode` | 通过机器人纯 X 轴平移，利用 `atan2(vy, vx)` 估计雷达安装 yaw 外参 `dpsi` |
| `icp_pcd_match.py` | 离线脚本 | 对两个 `.pcd` 文件做 ICP 点云配准，输出 4×4 变换矩阵、平移向量、ZYX 欧拉角，并可视化配准结果 |

---

## 各文件详细说明

---

### 1. `estimate_delay_by_gyro_corr.py`

**节点名**：`gyro_delay_estimator`

**功能**：通过比较 LIO 里程计（`/Odometry`）中的 `angular.z` 角速度与底盘消息（`/chassis_info`）中的 `wz + yaw_speed` 合并角速度，利用**归一化互相关（NCC）**在滑动时间窗内扫描不同时延，找到相关性最高的延迟值，从而估计两路信号之间的时间戳偏差。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `odom_topic` | `/Odometry` | LIO 里程计话题 |
| `chassis_topic` | `/chassis_info` | 底盘消息话题 |
| `window_sec` | `20.0` s | 用于计算相关的滑动窗口长度 |
| `lag_max_sec` | `0.5` s | 最大搜索时延范围 ±0.5 s |
| `lag_step_sec` | `0.002` s | 时延搜索步长（2 ms） |
| `resample_hz` | `200.0` Hz | 统一重采样频率 |
| `min_motion_radz` | `0.15` rad/s | 运动激励门限，低于此值的片段不参与计算 |
| `compute_period_sec` | `1.0` s | 计算周期 |

**输出**：通过 ROS log 打印 `best_delay_sec` 与 `best_corr`。

---

### 2. `extrinsic_test.py`

**节点名**：`chassis_to_mapped_odom`

**功能**：**在线外参验证/应用节点**。将底盘速度（`vx, vy, wz, yaw, yaw_speed`）通过几何外参模型变换到 LIO 雷达坐标系，融合 LIO 位姿后发布一路 `wheel_mapped_odom`，用于验证标定结果或作为 LIO 的辅助速度观测。

**核心运动学模型（SE2）**：

```
p_bl(φ) = p_bg + Rz(φ) · p_gl
v_l^b  = v_b + wz × p_bl + Rz(φ) · (φ̇ × p_gl)
R_bl   = Rz(φ + Δψ),  R_lb = R_bl^T
v_l^LIO = R_lb · v_l^b
ω_l    = wz + φ̇
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `p_gl_x/y` | `-0.082 / 0.168` m | 云台→雷达在云台系的平移 |
| `p_bg_x/y` | `0.0 / 0.0` m | 底盘→云台在底盘系的平移 |
| `dpsi_gl` | `0.0` rad | 雷达相对云台的偏航角偏差 |
| `chassis_topic` | `/chassis_info` | 底盘话题 |
| `odom_topic` | `/Odometry` | LIO 里程计话题 |
| `out_topic` | `/wheel_mapped_odom` | 输出里程计话题 |
| `max_lead_dt_sec` | `0.07` s | 底盘时间戳超前里程计的最大容忍量 |

**时间戳匹配规则**：为每条 LIO odom 消息选取时间戳刚好超前于 `t_odom + ODOM_DELAY_SEC`（默认 0.40 s）的最近底盘消息。

---

### 3. `offline_extrinsic_calib_se2.py`

**类型**：离线命令行脚本（不依赖 ROS 节点运行时）

**功能**：**离线标定核心工具**。从 rosbag2 读取 `/Odometry` 和 `/chassis_info`，对每对相邻里程计帧构建 SE(2) 相对位移残差（LIO 观测值 vs. 由底盘速度积分的预测值），通过 `scipy.optimize.least_squares`（Huber 损失）联合优化以下 5 个外参：

```
params = [p_gl_x, p_gl_y, p_bg_x, p_bg_y, Δψ]
```

**用法示例**：

```bash
python3 offline_extrinsic_calib_se2.py \
  --bag ~/bags/run1 \
  --odom_topic /Odometry \
  --chassis_topic /chassis_info \
  --pgl_x0 -0.082 --pgl_y0 0.168 \
  --pbg_x0 0.0 --pbg_y0 0.0 \
  --dpsi0 0.0
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bag` | 必填 | rosbag2 目录路径 |
| `--step_dt` | `0.01` s | 运动学积分步长 |
| `--min_dt / --max_dt` | `0.02 / 0.2` s | 有效里程计帧间隔范围 |
| `--motion_gate_w` | `0.02` rad | 最小转角门限（过滤静止片段） |
| `--huber` | `1.0` | Huber 损失尺度因子 |
| `--pgl_std / --pbg_std / --dpsi_std` | 软先验标准差，设为 inf 关闭 |

**输出**：打印优化后的外参值、代价函数值及收敛状态。

---

### 4. `P_gl_x_y.py`

**节点名**：`odom_offset_estimator`

**功能**：从 LIO `/Odometry` 的线速度和角速度实时估计雷达旋转中心偏移（即雷达相对机体旋转中心的偏移量 `p_gl`），将其转换到机体坐标系后在滑动窗口内求均值发布。

**原理**：利用刚体运动关系 `v = ω × r`，推导：

```
r_x^world =  v_y / ω_z
r_y^world = -v_x / ω_z
```

再将世界系偏移旋转到机体系输出。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_topic` | `/Odometry` | 输入里程计话题 |
| `window_sec` | `4.0` s | 均值估计时间窗 |
| `min_abs_omega` | `1e-3` rad/s | 角速度门限（过滤直行） |
| `vel_in_world` | `True` | 输入线速度是否为世界系 |

**发布话题**：
- `offset_xy` (`geometry_msgs/Vector3Stamped`)：机体系下估计的偏移 XY
- `radius` (`std_msgs/Float64`)：偏移量的模（旋转半径）

---

### 5. `P_gl_xy.py`

**节点名**：`lio_path_radius_fitter`

**功能**：订阅 LIO 发布的全局路径（`nav_msgs/Path`），对路径上的 XY 坐标点进行**Kåsa 代数圆拟合**（含截断最小二乘迭代去除异常点），输出拟合圆半径及拟合残差。可用于与 `P_gl_x_y.py` 的结果相互验证。

> **注意**：文件前半部分为已注释的旧版本（基于里程计速度统计平均半径），当前有效代码为 `LioPathRadiusFitter` 类。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_topic` | `/path` | LIO Path 话题 |
| `max_points` | `2000` | 最大缓存点数 |
| `trim_ratio` | `0.1` | 丢弃最大残差点比例（Trimmed LS） |
| `trim_iters` | `10` | 截断迭代次数 |
| `fit_rate_hz` | `0.5` Hz | 拟合触发频率 |

**发布话题**：
- `radius_fit` (`std_msgs/Float64`)：拟合圆半径
- `radius_fit_rmse` (`std_msgs/Float64`)：拟合 RMSE

---

### 6. `imu_calibration.py`

**节点名**：`imu_level_and_republish`

**功能**：对 IMU 原始数据进行**静态水平校正**。通过滑动窗口平均加速度计读数估计传感器的静态 roll/pitch 安装角（假设静止时 Z 轴朝上），构造旋转矩阵 `R = Rz(0)·Ry(pitch)·Rx(roll)` 将加速度和角速度旋转到水平坐标系，并重发布为新的 IMU 消息。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_topic` | `/livox/imu_192_168_1_133` | 原始 IMU 话题 |
| `out_topic` | `...leveled` | 校正后 IMU 话题 |
| `avg_window` | `100` 帧 | 平均窗口大小 |
| `use_avg_for_output_accel` | `False` | 输出加速度是否也用平均值 |
| `set_orientation` | `True` | 是否在输出 IMU 中填写由 roll/pitch 计算的 orientation |

---

### 7. `integrate_pointcloud_to_pcd.py`

**节点名**：`integrate_pointcloud_to_pcd`

**功能**：**点云积分存储工具**。订阅一个或多个 `PointCloud2` 话题，将指定帧数内的所有点云帧累积后，写出为标准 `.pcd` 文件。支持有/无 intensity 字段的点云，支持 ASCII 和 Binary 两种 PCD 格式。所有话题均完成后自动关闭节点。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `topics` | 两个 Livox 雷达话题 | 需要订阅的点云话题列表 |
| `frames` | `100` | 每个话题累积的帧数 |
| `out_dir` | `pcd_output/` | 输出目录 |
| `skip_nan` | `True` | 是否跳过 NaN 点 |
| `binary_pcd` | `True` | 是否以 Binary 格式存储 |

**输出文件命名**：`<topic_name_with_slash_replaced>_integrated_<N>frames.pcd`

---

### 8. `timestamp_test.py`

**节点名**：`topic_time_printer`

**功能**：**时间戳诊断工具**。同时订阅 `/Odometry` 和 `/chassis_info`，对每条收到的消息打印：节点接收时间、消息内嵌时间戳、两者之差（ms），用于快速排查消息延迟、时间同步问题。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `odom_topic` | `/Odometry` | 里程计话题 |
| `chassis_topic` | `/chassis_info` | 底盘话题 |
| `print_every_n` | `1` | 每 N 条消息打印一次 |
| `use_throttle_ms` | `0` | 打印节流间隔（ms），0 为不节流 |

---

### 9. `calib_lidar_yaw.py`

**节点名**：`lidar_yaw_calib`

**功能**：**雷达 Yaw 外参（`dpsi`）在线标定工具**。让机器人沿底盘 X 轴做纯直线平移（无转向），LIO 里程计输出的速度方向理论上应与底盘 X 轴对齐（`vy ≈ 0`）。若雷达存在 yaw 安装偏差 `dpsi`，则观测到：

```
vx_lio = V · cos(dpsi)
vy_lio = V · sin(dpsi)
→ dpsi = atan2(vy_lio, vx_lio)
```

节点对每帧计算 `dpsi` 样本，在滑动时间窗内做**循环均值（circular mean）**和循环标准差统计，并在退出时打印最终结果。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_topic` | `/Odometry` | LIO 里程计话题 |
| `window_sec` | `10.0` s | 滑动均值时间窗 |
| `min_speed_mps` | `0.15` m/s | 速度低于此值的样本丢弃（低速不可靠） |
| `max_abs_omega` | `0.05` rad/s | 角速度超过此值（机器人在转向）则丢弃 |
| `min_samples` | `20` | 窗内最少有效样本数 |
| `publish_rate_hz` | `10.0` Hz | 发布频率 |

**发布话题**：
- `~/dpsi_deg` (`std_msgs/Float64`)：偏航外参均值（度）
- `~/dpsi_rad` (`std_msgs/Float64`)：偏航外参均值（弧度）
- `~/dpsi_std_deg` (`std_msgs/Float64`)：循环标准差（度，反映样本一致性）

**操作步骤**：
1. 启动节点，控制机器人以 `≥0.15 m/s` 沿 X 轴方向直线行驶
2. 观察 `dpsi_std_deg` 收敛到较小值（< 1°）时读取 `dpsi_deg`
3. 按 Ctrl+C 退出，终端会打印最终标定结果
4. 将结果填入 `extrinsic_test.py` 的 `dpsi_gl` 参数或 `offline_extrinsic_calib_se2.py` 的 `--dpsi0` 初始值



```
1. 运行 timestamp_test.py
   → 确认 /Odometry 与 /chassis_info 的时间戳延迟情况

2. 运行 estimate_delay_by_gyro_corr.py
   → 精确估计 LIO odom 与底盘消息之间的时间偏差
   → 将结果填入 extrinsic_test.py 的 ODOM_DELAY_SEC

3. 控制机器人做匀速圆周运动，同时运行：
   - P_gl_x_y.py  → 实时估计旋转中心偏移（在线，粗标定）
   - P_gl_xy.py   → 基于 LIO Path 圆拟合验证旋转半径

4. 录制 rosbag，运行 offline_extrinsic_calib_se2.py
   → 精确优化 [p_gl_x, p_gl_y, p_bg_x, p_bg_y, Δψ] 五个外参

5. 将标定结果填入 extrinsic_test.py 参数，在线验证输出里程计质量

6. 若需采集点云用于手工验证，运行 integrate_pointcloud_to_pcd.py 保存累积点云，
   再用 icp_pcd_match.py 对两个雷达的点云做 ICP 配准验证外参

7. 若 IMU 有安装倾斜，先运行 imu_calibration.py 进行静态水平补偿
```

---

## 各文件详细说明（续）

---

### 10. `icp_pcd_match.py`

**类型**：离线命令行脚本（不依赖 ROS，直接用 Python 运行）

**功能**：输入两个 `.pcd` 文件，执行 **ICP（迭代最近点）点云配准**，将 source 点云对齐到 target 点云，输出：

- **4×4 变换矩阵** `T`（`T * source ≈ target`）
- **平移向量** `[tx, ty, tz]`（单位 m）
- **ZYX 欧拉角** `[roll, pitch, yaw]`（单位 deg 和 rad）
- ICP **fitness score** 与 **inlier RMSE**

并弹出 Open3D 可视化窗口：
- 🟢 **绿色**：source 原始位置
- 🔵 **蓝色**：source ICP 配准后
- 🔴 **红色**：target（参考点云）

**配准流程**：
1. 体素降采样（可选）
2. FPFH 全局粗配准（RANSAC，可选）→ 给 ICP 提供可靠初值，适合两点云初始位姿差异较大的情况
3. ICP 精配准（支持 point-to-point / point-to-plane / generalized 三种方法）

**用法**：

```bash
# 基本用法（point-to-point ICP + 自动全局粗配准）
python3 icp_pcd_match.py source.pcd target.pcd

# 使用 point-to-plane ICP，自定义体素大小和对应距离
python3 icp_pcd_match.py source.pcd target.pcd \
    --method point_to_plane --voxel 0.05 --dist 0.1

# 已知初始位置接近，跳过全局配准，不弹可视化
python3 icp_pcd_match.py source.pcd target.pcd --no_global --no_vis

# 指定自定义初始变换矩阵（行优先 16 个数）
python3 icp_pcd_match.py source.pcd target.pcd \
    --init_T 1 0 0 0.1  0 1 0 0.05  0 0 1 0  0 0 0 1
```

**命令行参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `source` | 必填 | 待配准点云路径（`.pcd`） |
| `target` | 必填 | 参考点云路径（`.pcd`） |
| `--method` | `point_to_point` | ICP 方法：`point_to_point` / `point_to_plane` / `generalized` |
| `--voxel` | `0.05` m | 体素大小，`0` 表示不降采样 |
| `--dist` | `0.5` m | ICP 最大对应点距离阈值 |
| `--max_iter` | `200` | ICP 最大迭代次数 |
| `--no_global` | - | 跳过全局粗配准（RANSAC） |
| `--no_vis` | - | 不弹出可视化窗口 |
| `--init_T` | - | 自定义初始变换矩阵（行优先 16 个数） |

**依赖**（与 ROS 无关，独立安装）：
```bash
pip install open3d numpy scipy
```

---

## 依赖

- ROS 2 (Humble 或更高)
- `rclpy`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`, `std_msgs`
- `robot_msg`（本仓库自定义消息包，含 `ChassisMsg`）
- `numpy`, `scipy`（用于 `offline_extrinsic_calib_se2.py` 和 `icp_pcd_match.py`）
- `rosbag2_py`, `rclpy.serialization`（用于 `offline_extrinsic_calib_se2.py`）
- `sensor_msgs_py`（用于 `integrate_pointcloud_to_pcd.py`）
- `open3d`（仅用于 `icp_pcd_match.py`，`pip install open3d`）

