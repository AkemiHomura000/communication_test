# Small Point-LIO 项目架构文档

## 1. 目录结构

```
small_point_lio/
├── src/
│   ├── small_point_lio_node.cpp/hpp      # ROS2 节点包装层（对外接口）
│   ├── common/
│   │   └── common.h                      # 公共数据结构（Point / ImuMsg / Odometry）
│   ├── small_point_lio/                  # 核心算法模块
│   │   ├── small_point_lio.h/.cpp        # 主算法入口，协调预处理与估计
│   │   ├── estimator.h/.cpp              # 测量模型（点云 / IMU）
│   │   ├── eskf.h                        # 误差状态卡尔曼滤波器（ESKF）
│   │   ├── preprocess.h/.cpp             # 传感器数据缓冲与点云过滤
│   │   ├── parameters.h/.cpp             # ROS2 参数加载与管理
│   │   ├── small_ivox.h                  # 增量式体素地图（哈希 + LRU）
│   │   └── so3_math.h                    # SO(3) 李群数学工具
│   ├── lidar_adapter/                    # 雷达数据适配层（策略模式）
│   │   ├── base_lidar.h                  # 抽象接口
│   │   ├── livox_custom_msg.h            # Livox 自定义格式
│   │   ├── livox_pointcloud2.h           # Livox PointCloud2 格式
│   │   ├── custom_mid360_driver.h        # 自定义 Mid360 驱动格式
│   │   └── unitree_lidar.h              # 宇树雷达格式
│   ├── util/
│   │   ├── voxelgrid_sampling.h/.cpp     # 体素下采样（空间滤波）
│   │   └── pointcloud_mapping.h/.cpp     # 地图点云积累（PCD 保存用）
│   └── io/
│       └── pcd_io.h/.cpp                 # PCD 文件读写
├── config/                               # YAML 参数配置文件
├── launch/                               # ROS2 launch 文件
└── include/
    ├── pch.h                             # 预编译头
    └── param_deliver.h.in                # CMake 路径注入模板
```

---

## 2. 核心数据结构

### 2.1 `state`（eskf.h）
30 维 ESKF 状态向量，所有字段均为 `double` 类型：

| 字段 | 维度 | 索引起始 | 含义 |
|------|------|---------|------|
| `position` | 3 | 0 | 世界坐标系位置 |
| `rotation` | 3×3 | 3 | SO(3) 旋转矩阵（IMU body → world） |
| `offset_R_L_I` | 3×3 | 6 | 雷达到 IMU 的旋转外参 |
| `offset_T_L_I` | 3 | 9 | 雷达到 IMU 的平移外参 |
| `velocity` | 3 | 12 | 世界坐标系速度 |
| `omg` | 3 | 15 | IMU 体坐标系角速度 |
| `acceleration` | 3 | 18 | IMU 体坐标系加速度 |
| `gravity` | 3 | 21 | 重力向量（世界坐标系） |
| `bg` | 3 | 24 | 陀螺仪零偏 |
| `ba` | 3 | 27 | 加速度计零偏 |

`state::plus(δx)` 在流形上做更新：旋转量使用 `exp(δθ)` 左乘，其余量直接加法。

### 2.2 `point_measurement_result`（eskf.h）

| 字段 | 类型 | 含义 |
|------|------|------|
| `valid` | bool | 是否找到有效平面 |
| `z` | scalar | 点到平面的有符号距离（残差） |
| `H` | 1×12 | 对 [position, rotation, offset_R_L_I, offset_T_L_I] 的雅可比 |
| `laser_point_cov` | scalar | 激光点噪声协方差 |

### 2.3 `imu_measurement_result`（eskf.h）

| 字段 | 类型 | 含义 |
|------|------|------|
| `z` | 6×1 | 残差：[ω_meas − omg − bg; a_meas·scale − acc − ba] |
| `satu_check[6]` | bool[6] | 各轴饱和检测标志（饱和则置零残差） |
| `imu_meas_omg_cov` | scalar | 陀螺仪测量噪声 |
| `imu_meas_acc_cov` | scalar | 加速度计测量噪声 |

### 2.4 `common::Point` / `ImuMsg` / `Odometry`（common.h）
传感器与输出消息基础结构，均含 `timestamp` 字段（秒，浮点）。

### 2.5 `SmallIVox`（small_ivox.h）
基于哈希表 + LRU 链表的增量式体素地图：
- `grids_map`：`uint64_t hash → list<Vector3f>::iterator`（O(1) 查找）
- 哈希键由 `uint16_t [x, y, z]` 体素坐标编码为 64 位整数
- 搜索时访问中心格 + 6 方向面邻格（共 7 个候选）
- 超出容量时自动驱逐 LRU 最久未访问的体素格

---

## 3. 类关系图

```
SmallPointLioNode（ROS2 Component）
│
├── SmallPointLio                    ← 核心算法主控
│   ├── Parameters                   ← 所有配置参数
│   ├── Preprocess                   ← 数据缓冲与过滤
│   │   ├── point_deque              （降采样后点，用于 ESKF 更新）
│   │   ├── dense_point_deque        （全量点，用于点云积累）
│   │   └── imu_deque                （IMU 消息队列）
│   └── Estimator                    ← 测量模型与地图管理
│       ├── eskf kf                  （ESKF 状态机）
│       └── SmallIVox ivox           （增量体素地图）
│
├── BaseLidarAdapter（多态）          ← 雷达数据适配
│   ├── LivoxCustomMsgAdapter
│   ├── LivoxPointCloud2Adapter
│   ├── CustomMid360DriverAdapter
│   └── UnilidarAdapter
│
├── tf2_ros::TransformBroadcaster    ← 发布 lidar_odom → base_link TF
├── tf2_ros::Buffer + TransformListener  ← 查询静态外参 TF
├── Publisher<Odometry>              → /Odometry
├── Publisher<PointCloud2>           → /cloud_registered
├── Service<Trigger>                 → map_save
└── util::PointcloudMapping          ← 可选，地图点积累 → PCD 文件
```

---

## 4. 主处理流程

### 4.1 初始化阶段

```
传感器数据到达 → on_point_cloud_callback() / on_imu_callback()
                        ↓
               Preprocess 缓存数据
                        ↓
          handle_once() 检查初始化条件：
          point_deque.size() >= init_map_size
          && (imu_deque.size() >= 200 if fix_gravity)
                        ↓
          for each point in point_deque:
              ivox->add_point(point)      ← 建初始地图
                        ↓
          从 imu_deque 均值估计重力方向
                        ↓
          kf.init_timestamp(time_current)
          is_init = true
```

### 4.2 主循环（Point-LIO 融合）

`handle_once()` 每次传感器回调都调用，按时间戳优先级消费三路队列：

```
while (imu_deque && dense_point_deque && point_deque 非空):
│
├─ [dense_point 最早] → 仅积累点云，不更新滤波器
│     p_imu = R_LI * p_lidar + T_LI
│     p_world = R_world * p_imu + t_world
│     → 写入 pointcloud_odom_frame
│
├─ [point 最早] → ESKF 点云更新
│     kf.predict_state(t)
│         x.position += velocity * dt
│         x.rotation *= exp(omg * dt)
│         x.velocity += (R * acc + gravity) * dt
│     ↓
│     Estimator::h_point()
│         transform point: lidar → IMU → world
│         ivox->get_closest_point() → 7格邻域搜索取5近邻
│         PCA 协方差特征值分解 → 平面法向量 n, 截距 d
│         验证平面质量：所有近邻点距离 < plane_threshold
│         验证当前点：距离比例 < match_sqaured
│         计算残差 z = -(n·p_world + d)
│         计算雅可比 H (1×12)
│     kf.update_point()：K = PHᵀ / (HPHᵀ + R)
│                         x += K·z，P -= K·H·P
│     ↓
│     publish_odometry()（按 odometry_publish_rate 限频）
│     ivox->add_point(point_odom_frame)  ← 增量更新地图
│
└─ [IMU 最早] → ESKF IMU 更新
      kf.predict_state(t)
      kf.predict_cov(t, Q)
          F = 状态转移矩阵（含 SO(3) 切线空间线性化）
          P = F P Fᵀ + Q·dt²
      ↓
      Estimator::h_imu()
          z[0:3] = ω_meas − omg − bg
          z[3:6] = a_meas·scale − acc − ba
          （饱和轴置零）
      kf.update_imu()：用 LDLT 分解求 Kalman 增益
                        x += K·z，P -= K·H·P
```

### 4.3 输出发布

```
pointcloud_callback（点云积累完成时触发）
    ↓ 变换到 base_link 坐标系
    → 发布 /cloud_registered (sensor_msgs/PointCloud2)
    → 可选：pointcloud_mapping->add_point() 积累用于保存 PCD

odometry_callback（每次 publish_odometry 调用时触发）
    ↓ TF 查询 lidar_frame → base_link 静态外参
    ↓ 计算 odom → base_link 变换
       T_odom_base = T_base_lidar⁻¹ · T_odom_lidar · T_base_lidar
    ↓ 速度变换（杆臂补偿）
       omega_base   = R_lidar_to_base · omega_lidar
       v_base_world = v_lidar_world − R_lidar_world · (omega_lidar × r_lidar_to_base)
    → TF 广播 lidar_odom → base_link
    → 发布 /Odometry (nav_msgs/Odometry)
```

---

## 5. 关键参数说明

| 参数名 | 类型 | 含义 |
|--------|------|------|
| `point_filter_num` | int | 每隔 N 个点取 1 个（时间降采样）|
| `min/max_distance` | float | 有效点距离范围（米）|
| `space_downsample` | bool | 是否启用体素空间下采样 |
| `space_downsample_leaf_size` | float | 下采样体素大小（米）|
| `map_resolution` | float | 地图体素分辨率（米）|
| `init_map_size` | int | 初始化所需累积点数 |
| `fix_gravity_direction` | bool | 是否从 IMU 均值估计重力方向 |
| `extrinsic_est_en` | bool | 是否在线估计 LiDAR-IMU 外参 |
| `odometry_publish_rate` | double | 里程计发布频率（Hz，默认 200）|
| `plane_threshold` | double | 平面有效性验证距离阈值（米）|
| `match_sqaured` | double | 点到平面距离比例阈值（离群点剔除）|
| `laser_point_cov` | double | 激光点测量噪声协方差 |
| `imu_meas_omg/acc_cov` | double | IMU 陀螺仪 / 加速度计测量噪声 |
| `velocity/omg/acceleration_cov` | double | 过程噪声协方差（对应状态维度）|
| `bg/ba_cov` | double | 零偏随机游走噪声协方差 |

---

## 6. 支持的雷达类型

| `lidar_type` 参数值 | 适配器类 | 说明 |
|---------------------|---------|------|
| `livox_custom_msg` | `LivoxCustomMsgAdapter` | 需编译时启用 `HAVE_LIVOX_DRIVER` |
| `livox_pointcloud2` | `LivoxPointCloud2Adapter` | Livox 标准 PointCloud2 |
| `custom_mid360_driver` | `CustomMid360DriverAdapter` | 自定义 Mid360 驱动 |
| `unilidar` | `UnilidarAdapter` | 宇树 L1/L2 雷达 |
