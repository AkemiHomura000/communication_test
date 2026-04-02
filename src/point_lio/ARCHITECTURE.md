# Point-LIO 架构文档：类与函数调用关系

> 版本：Phase 6（多模块封装完成）  
> 仅保留 `state_output` / `kf_output` 输出模式，已移除 IMU-as-input 分支。

---

## 一、模块总览

```
main.cpp
  └─► LaserMappingNode          (rclcpp::Node 子类，主控节点)
        ├─► Estimator            (EKF 状态 + 点云缓冲区 + 观测模型)
        │     └─► esekfom::esekf (第三方 IKFoM 库，通过函数指针回调)
        └─► LidarImuBuffer       (传感器数据缓冲 + ROS 回调 + 时间同步)

parameters.h / parameters.cpp   (全局配置参数，由 readParameters() 填充)
```

---

## 二、各文件职责

| 文件 | 类/内容 | 职责 |
|---|---|---|
| `main.cpp` | — | 进程入口；500 Hz 驱动循环 |
| `LaserMappingNode.h/.cpp` | `LaserMappingNode` | ROS 节点主控，持有 Estimator 和 LidarImuBuffer |
| `Estimator.h/.cpp` | `Estimator` | EKF 实例、点云工作缓冲区、观测模型实现 |
| `li_initialization.h/.cpp` | `LidarImuBuffer` | LiDAR/IMU ROS 回调、时间同步、数据帧队列 |
| `parameters.h/.cpp` | (全局变量) | 所有配置参数的声明与从 ROS 参数服务器读取 |

---

## 三、`main.cpp` — 程序入口

```
main()
  ├── rclcpp::init()
  ├── make_shared<LaserMappingNode>()        →  LaserMappingNode::LaserMappingNode()
  ├── node->postInit()                       →  LaserMappingNode::postInit()
  ├── signal(SIGINT, SigHandle)
  └── 500 Hz 循环
        ├── executor.spin_some()             →  触发 ROS 回调（见第四节）
        └── node->spin_once()               →  LaserMappingNode::spin_once()
```

**全局变量**
- `bool g_flg_exit` — 退出标志，`SigHandle` 设置，主循环检查。

---

## 四、`LidarImuBuffer` — 传感器数据缓冲

### 4.1 ROS 回调（由 executor.spin_some() 触发）

```
ROS 话题 /livox/lidar  (Livox 格式)
  └── sub_pcl_livox_ lambda
        └── LidarImuBuffer::livox_pcl_cbk()
              ├── p_pre->process()            预处理/去畸变
              └── pushLidarFrame()            写入 lidar_buffer / time_buffer

ROS 话题 /livox/lidar  (标准 PointCloud2)
  └── sub_pcl_pc_ lambda
        └── LidarImuBuffer::standard_pcl_cbk()
              ├── p_pre->process()
              └── pushLidarFrame()

ROS 话题 /livox/imu
  └── sub_imu_ lambda
        └── LidarImuBuffer::imu_cbk()
              └── imu_deque.push_back()       带时间偏移补偿
```

### 4.2 帧合并与切割

**`cut_frame` 模式**（在回调中直接处理，不经过 `pushLidarFrame`）：
```
standard_pcl_cbk() / livox_pcl_cbk()
  └── [cut_frame_init=true]
        p_pre->process_cut_frame_pcl2()   将一帧按时间切成 cut_frame_num 段
        └── 每段直接 push_back 进 lidar_buffer / time_buffer
```

**`con_frame` 模式**（帧合并，经过 `pushLidarFrame`）：
```
LidarImuBuffer::pushLidarFrame(ptr, timestamp)   [私有]
  └── [con_frame=true]
        ├── frame_ct_==0 时记录起始时间 time_con = last_timestamp_lidar
        ├── frame_ct_ < 10：
        │     ├── 将每个点的 curvature += (last_timestamp_lidar - time_con)*1000
        │     │   （把绝对时间戳转为相对帧内偏移，单位 ms）
        │     └── 追加到内部缓冲 ptr_con_，frame_ct_++
        └── frame_ct_ == 10：
              ├── 将合并好的 ptr_con_ 推入 lidar_buffer / time_buffer
              └── 清空 ptr_con_，frame_ct_ = 0，开始下一轮

  [con_frame=false]（默认，逐帧直推）
        └── 直接 emplace_back 到 lidar_buffer / time_buffer
```

> **两种模式的目的**
> - `cut_frame`：把一帧稀疏扫描切成多段，提高 KF 更新频率（适合机械式激光雷达）。
> - `con_frame`：把连续多帧合并成一帧，增加单次 KF 可用点数（适合点少的 Livox 等固态雷达）。
> - 两者互斥，默认均关闭（逐帧直推）。

### 4.3 时间同步

```
LidarImuBuffer::syncPackages(MeasureGroup &meas)   ← LaserMappingNode::spin_once() 调用
  ├── 检查 lidar_buffer / imu_deque 是否非空
  ├── 匹配 LiDAR 帧时间范围内的所有 IMU 数据
  ├── 填充 meas.lidar, meas.imu, meas.lidar_beg_time
  └── 返回 true = 本帧数据就绪
```

**公开数据成员（LaserMappingNode 直接访问）**

| 成员 | 用途 |
|---|---|
| `lidar_buffer` | LiDAR 帧队列 |
| `time_buffer` | 对应时间戳队列 |
| `imu_deque` | IMU 消息队列 |
| `imu_last / imu_next` | 当前处理窗口的前后 IMU 帧 |
| `scan_count` | 累计帧数（用于日志数组索引） |
| `T1[], s_plot[]` 等 | 调试用时序数组 |

---

## 五、`Estimator` — EKF 状态与观测模型

### 5.1 初始化链

```
LaserMappingNode::initKalmanFilter()
  ├── reset_cov_output(P_init_output_)          参数.cpp 中定义
  ├── Estimator::process_noise_cov_output()     [static] 返回 Q 矩阵
  └── estimator_.initKalmanFilter(P, Q)
        └── kf_output.init_dyn_share_modified_3h(
              get_f_output,             [static] 状态转移函数
              df_dx_output,             [static] 状态转移 Jacobian
              h_model_output_bridge,    [自由函数桥接] → h_model_output()
              h_model_IMU_output_bridge [自由函数桥接] → h_model_IMU_output()
            )
```

### 5.2 桥接机制（Bridge Pattern）

`esekfom::esekf` 只接受自由函数指针，但观测模型需要访问 `Estimator` 成员数据：

```
g_estimator = &estimator_    （LaserMappingNode 构造函数设置）

h_model_output_bridge(s, cov_p, cov_R, ekfom_data)
  └── g_estimator->h_model_output(s, cov_p, cov_R, ekfom_data)
        ├── ivox_->GetClosestPoint()    最近邻搜索
        ├── 平面拟合，计算法向量
        └── 填充 ekfom_data (H 矩阵, z 残差)

h_model_IMU_output_bridge(s, ekfom_data)
  └── g_estimator->h_model_IMU_output(s, ekfom_data)
        └── 利用 angvel_avr / acc_avr 构建 IMU 观测残差
```

### 5.3 坐标变换

```
Estimator::pointBodyToWorld(pi, po)
  ├── [extrinsic_est_en=true]  使用 kf_output.x_.offset_R/T_L_I（在线估计外参）
  └── [extrinsic_est_en=false] 使用固定 Lidar_R/T_wrt_IMU

LaserMappingNode::pointBodyLidarToIMU(pi, po)   [static，通过 g_estimator 访问]
  ├── [extrinsic_est_en=true]  kf_output.x_.offset_R/T_L_I
  └── [extrinsic_est_en=false] Lidar_R/T_wrt_IMU
```

**公开数据成员（LaserMappingNode 直接访问）**

| 成员 | 用途 |
|---|---|
| `kf_output` | EKF 实例，内含状态 `x_`（pos/vel/rot/omg/acc/bg/ba/gravity/offset_*） |
| `input_in` | EKF 输入（当前帧角速度/加速度） |
| `feats_down_body` | 下采样后的 body 系点云 |
| `feats_down_world` | 转换到世界系后的点云 |
| `normvec` | 法向量工作缓冲 |
| `Nearest_Points` | 每个点的最近邻结果 |
| `ivox_` | IVox 增量体素地图 |
| `time_seq` | 逐点时序分组 |
| `k, idx` | 逐点迭代游标 |
| `angvel_avr / acc_avr` | 当前 IMU 测量均值 |

---

## 六、`LaserMappingNode` — 主控节点

### 6.1 构造与初始化

```
LaserMappingNode::LaserMappingNode()
  ├── g_estimator        = &estimator_       设置全局单例（桥接用）
  ├── g_lidar_imu_buffer = &lidar_imu_buf_   设置全局单例（备用）
  ├── initParameters()
  │     ├── readParameters()                 从 ROS 参数服务器读取所有配置
  │     ├── estimator_.ivox_ = make_shared<IVoxType>()
  │     ├── estimator_.Lidar_T/R_wrt_IMU    外参初始化
  │     └── odom_pub_interval_               里程计发布间隔计算
  ├── initKalmanFilter()                     见第五节 5.1
  ├── initLogFiles()                         打开调试日志文件
  ├── initSubscribers()                      注册 LiDAR/IMU ROS 订阅者
  ├── initPublishers()                       注册点云/里程计/轨迹发布者
  └── initServiceServer()                    注册 /reset_map_and_odom 服务

LaserMappingNode::postInit()               由 main() 在 make_shared 后调用
  └── 初始化 tf_broadcaster_ / tf_buffer_ / tf_listener_  （需要 shared_from_this()）
```

### 6.2 `spin_once()` — 主循环（500 Hz 调用）

```
spin_once()
  ├── lidar_imu_buf_.syncPackages(Measures)  → false: 本帧跳过
  │
  ├── [flg_reset_]  handleReset()
  │     ├── p_imu->Reset()
  │     ├── kf_output.change_P()
  │     └── ivox_.reset()
  │
  ├── [flg_first_scan_]  handleFirstScan()
  │     ├── 记录 first_lidar_time
  │     └── 初始化 gravity / imu_deque 头部对齐
  │
  ├── t0 ─┐
  │       │ downsampleAndSort()
  │       │   ├── p_imu->Process()           IMU 积分去畸变
  │       │   ├── downSizeFilterSurf_.filter()  体素下采样
  │       │   ├── sort() by time_list
  │       │   ├── time_compressing()         生成 time_seq
  │       │   └── feats_down_size 更新
  │ t1 ─┘
  │
  ├── [!p_imu->after_imu_init_]  tryImuInit()
  │     ├── p_imu->Set_init()               重力方向对齐
  │     └── kf_output.x_.rot 初始化
  │
  ├── [!init_map_]  buildInitMap()
  │     ├── pointBodyToWorld() × N
  │     ├── 累积至 init_feats_world_
  │     ├── [达到 init_map_size]  ivox_->AddPoints() + publishInitMap()
  │     └── 返回 false（本帧跳过后续 KF）
  │
  ├── preparePointLists()
  │     └── 预计算 pbody_list / crossmat_list
  │
  ├── t3 ─┐
  │       │ [time_seq 非空]  runPointByPointUpdate()     ←─见 6.3
  │       │ [time_seq 为空]  runImuOnlyUpdate()          ←─见 6.4
  │ t5 ─┘
  │
  ├── [feats_down_size > 4]  MapIncremental()
  │     └── ivox_->AddPoints(points_to_add)
  │
  ├── [path_en]         publishPath()
  ├── [scan_pub_en]     publishFrameWorld()
  ├── [scan_body_pub_en]publishFrameBody()
  │
  ├── 1 Hz 性能日志（spin_once 耗时统计）
  └── [runtime_pos_log] logRuntimeStats()
```

### 6.3 `runPointByPointUpdate()` — 逐点 KF 更新

```
runPointByPointUpdate()
  └── for k in time_seq:
        ├── time_current = point_body.curvature/1000 + pcl_beg_time
        │
        ├── [is_first_frame]  alignImuToFirstPoint(imu_upda_cov)
        │     └── 跳过早于当前时刻的 IMU，记录 angvel/acc_avr，标记 is_first_frame=false
        │
        ├── [imu_en && imu_deque 非空]  processImuBeforePoint(imu_upda_cov)
        │     ├── 消耗早于 time_current 的 IMU 帧
        │     ├── kf_output.predict()    × 每个 IMU 步
        │     └── kf_output.update_iterated_dyn_share_IMU()  IMU 观测更新
        │
        ├── propagateState()
        │     └── kf_output.predict(dt, Q, input_in)   传播到当前激光点时刻
        │
        ├── kf_output.update_iterated_dyn_share_modified()
        │     └── → h_model_output_bridge → Estimator::h_model_output()
        │           ├── ivox_->GetClosestPoint()
        │           └── 平面残差 + H 矩阵 → EKF 更新
        │
        ├── [达到 odom_pub_interval_]  publishOdometry()
        │     ├── setPoseStamp() / setTwist()  从 kf_output.x_ 读取
        │     ├── pub_odom_->publish()
        │     └── publishTfToBaseLink()
        │           ├── tf_buffer_->lookupTransform(livox_frame → base_link)
        │           └── tf_broadcaster_->sendTransform(lidar_odom → base_link)
        │
        └── pointBodyToWorld() × time_seq[k]  更新 feats_down_world
```

### 6.4 `runImuOnlyUpdate()` — 纯 IMU 传播帧

```
runImuOnlyUpdate()
  └── while imu_next.stamp < lidar_beg_time + lidar_time_inte:
        ├── kf_output.predict(dt_cov, Q, input_in, false, true)  协方差传播
        ├── kf_output.predict(dt,     Q, input_in, true,  false) 均值传播
        └── kf_output.update_iterated_dyn_share_IMU()
              └── → h_model_IMU_output_bridge → Estimator::h_model_IMU_output()
```

### 6.5 发布函数汇总

| 函数 | 触发条件 | 发布话题 |
|---|---|---|
| `publishInitMap()` | `buildInitMap()` 完成 | `/Laser_map` |
| `publishFrameWorld()` | `scan_pub_en \|\| pcd_save_en` | `/cloud_registered` |
| `publishFrameBody()` | `scan_body_pub_en` | `/cloud_registered_body` |
| `publishOdometry()` | 逐点更新中频率门控 | `/Odometry` |
| `publishTfToBaseLink()` | 随 `publishOdometry()` 调用 | TF: `lidar_odom→base_link` |
| `publishPath()` | `path_en` | `/path` |

### 6.6 日志函数

```
logRuntimeStats(t0, t1, t3, t5)          每帧调用（runtime_pos_log=true）
  ├── 更新 aver_time_* 滑动平均
  ├── 写入 lidar_imu_buf_.T1[] / s_plot[]
  ├── 写入 fout_out（CSV 状态日志）
  └── dumpLioStateToLog()                 写入 fp_（二进制位置日志）

logPbpState()                             逐点更新中每次 KF 更新后调用
  └── 写入 fout_out
```

---

## 七、全局单例与桥接关系

```
LaserMappingNode 构造函数
  ├── g_estimator        = &estimator_        (Estimator*)
  └── g_lidar_imu_buffer = &lidar_imu_buf_    (LidarImuBuffer*)

esekfom::esekf 注册的函数指针:
  get_f_output          [Estimator static]   状态转移 f(x, u)
  df_dx_output          [Estimator static]   ∂f/∂x
  h_model_output_bridge [自由函数]            → g_estimator->h_model_output()
  h_model_IMU_output_bridge [自由函数]        → g_estimator->h_model_IMU_output()
```

---

## 八、参数流向

```
ROS 参数服务器
  └── readParameters(node)               parameters.cpp
        └── 填充所有全局变量（lid_topic, filter_size_*, imu_en, ...）

LaserMappingNode::initParameters()
  ├── 读取参数 → estimator_.Lidar_T/R_wrt_IMU
  ├── 读取参数 → estimator_.kf_output.x_.offset_*
  └── 读取参数 → odom_pub_interval_

LaserMappingNode::initKalmanFilter()
  └── reset_cov_output() + process_noise_cov_output() → estimator_.initKalmanFilter()
```

---

## 九、依赖关系图（头文件包含）

```
main.cpp
  └── LaserMappingNode.h
        ├── Estimator.h
        │     ├── common_lib.h      (PointType, V3D, M3D, state_output, ...)
        │     └── parameters.h      (IVoxType, 全局变量声明)
        ├── li_initialization.h
        │     └── Estimator.h
        └── common_lib.h

parameters.h
  ├── preprocess.h    (Preprocess 类, p_pre)
  ├── IMU_Processing.h (ImuProcess 类, p_imu)
  └── ivox/ivox3d.h   (IVox 类模板)
```

---

## 十、关键数据流（一帧完整路径）

```
LiDAR 原始数据
  →[ROS 回调] livox_pcl_cbk / standard_pcl_cbk
  →[缓冲] lidar_buffer
  →[同步] syncPackages() → Measures.lidar / Measures.lidar_beg_time
  →[去畸变] p_imu->Process() → feats_undistort_
  →[下采样] downSizeFilterSurf_ → feats_down_body
  →[时序] time_compressing() → time_seq

IMU 数据
  →[ROS 回调] imu_cbk() → imu_deque
  →[同步] syncPackages() → Measures.imu
  →[积分] p_imu->Process() (去畸变用)
  →[逐点传播] processImuBeforePoint() / propagateState()
    → kf_output.predict()

EKF 更新
  →[逐点] kf_output.update_iterated_dyn_share_modified()
    → h_model_output() → ivox_->GetClosestPoint() → 平面残差
    → kf_output.x_ 更新（pos / vel / rot / bg / ba / ...）

输出
  →[发布] publishOdometry() → /Odometry (nav_msgs::Odometry)
  →[发布] publishTfToBaseLink() → TF lidar_odom→base_link
  →[地图] MapIncremental() → ivox_->AddPoints()
  →[点云] publishFrameWorld() → /cloud_registered
```
