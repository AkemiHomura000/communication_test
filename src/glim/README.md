# glim

LiDAR-IMU SLAM based on GTSAM and gtsam_points.

---

## Installation

### 1. System Dependencies

```bash
sudo apt install libomp-dev libboost-all-dev libmetis-dev \
                 libfmt-dev libspdlog-dev \
                 libglm-dev libglfw3-dev libpng-dev libjpeg-dev
```

### 2. GTSAM

```bash
git clone https://github.com/borglab/gtsam
cd gtsam && git checkout 4.3a0
mkdir build && cd build
cmake .. -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
         -DGTSAM_BUILD_TESTS=OFF \
         -DGTSAM_WITH_TBB=OFF \
         -DGTSAM_USE_SYSTEM_EIGEN=ON \
         -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF
make -j$(nproc)
sudo make install
```

### 3. Iridescence (optional, recommended for 3D viewer)

```bash
git clone https://github.com/koide3/iridescence --recursive
mkdir iridescence/build && cd iridescence/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### 4. gtsam_points

```bash
git clone https://github.com/koide3/gtsam_points
mkdir gtsam_points/build && cd gtsam_points/build
cmake .. -DBUILD_WITH_CUDA=ON
make -j$(nproc)
sudo make install
```

### 5. Update shared library cache

```bash
sudo ldconfig
```

---

## Build (ROS 2)

```bash
cd ~/your_ws
colcon build --cmake-args -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_VIEWER=ON -DBUILD_WITH_MARCH_NATIVE=OFF
source install/setup.bash
```

> If Iridescence is not installed, set `-DBUILD_WITH_VIEWER=OFF`.

---

## Quick Start

```bash
# Terminal 1 — run glim
ros2 run glim_ros glim_rosnode --ros-args -p config_path:=/path/to/glim/config

# Terminal 2 — play rosbag
ros2 bag play /path/to/rosbag --topics /livox/lidar_xxx /livox/imu_xxx

# Terminal 3 — visualize
rviz2 -d src/glim/glim_ros2/rviz/glim_ros.rviz
```

> `config_path` must point to the **directory** containing `config.json`, not the file itself.

---

## Published Topics

All topics are published under the node namespace (default: `/glim_rosnode/`).

### Odometry (IMU frame)

| Topic | Type | Description |
|-------|------|-------------|
| `~/odom` | `nav_msgs/Odometry` | IMU frame odometry at scan start time |
| `~/odom_scanend` | `nav_msgs/Odometry` | IMU frame odometry at scan end time |
| `~/odom_corrected` | `nav_msgs/Odometry` | Loop-closure corrected odometry |
| `~/odom_scanend_corrected` | `nav_msgs/Odometry` | Loop-closure corrected odometry at scan end |

### Odometry (LiDAR frame)

| Topic | Type | Description |
|-------|------|-------------|
| `~/lidar_odom` | `nav_msgs/Odometry` | LiDAR frame odometry at scan start time |
| `~/lidar_odom_scanend` | `nav_msgs/Odometry` | LiDAR frame odometry at scan end time |
| `~/lidar_odom_corrected` | `nav_msgs/Odometry` | Loop-closure corrected lidar odometry |
| `~/lidar_odom_scanend_corrected` | `nav_msgs/Odometry` | Loop-closure corrected lidar odometry at scan end |

### Pose (IMU frame)

| Topic | Type | Description |
|-------|------|-------------|
| `~/pose` | `geometry_msgs/PoseStamped` | IMU frame pose at scan start time |
| `~/pose_scanend` | `geometry_msgs/PoseStamped` | IMU frame pose at scan end time |
| `~/pose_corrected` | `geometry_msgs/PoseStamped` | Loop-closure corrected pose |
| `~/pose_scanend_corrected` | `geometry_msgs/PoseStamped` | Loop-closure corrected pose at scan end |

### Pose (LiDAR frame)

| Topic | Type | Description |
|-------|------|-------------|
| `~/lidar_pose` | `geometry_msgs/PoseStamped` | LiDAR frame pose at scan start time |
| `~/lidar_pose_scanend` | `geometry_msgs/PoseStamped` | LiDAR frame pose at scan end time |
| `~/lidar_pose_corrected` | `geometry_msgs/PoseStamped` | Loop-closure corrected lidar pose |
| `~/lidar_pose_scanend_corrected` | `geometry_msgs/PoseStamped` | Loop-closure corrected lidar pose at scan end |

### Point Clouds

| Topic | Type | Description |
|-------|------|-------------|
| `~/points` | `sensor_msgs/PointCloud2` | Raw input point cloud per scan |
| `~/aligned_points` | `sensor_msgs/PointCloud2` | Scan-to-map aligned point cloud |
| `~/points_corrected` | `sensor_msgs/PointCloud2` | Loop-closure corrected input cloud |
| `~/aligned_points_corrected` | `sensor_msgs/PointCloud2` | Loop-closure corrected aligned cloud |
| `~/map` | `sensor_msgs/PointCloud2` | Full global map (merged submaps) |

### TF

| Transform | Description |
|-----------|-------------|
| `map → odom → imu_frame` | Published per odometry frame |
| `imu_frame → lidar_frame` | Published once (static), controlled by `publish_imu2lidar` |

---

## Publish Logic

- **Frequency**: All per-frame topics (`odom`, `pose`, `points`, etc.) are published **once per LiDAR scan** (same rate as the point cloud, typically 10 Hz for Livox sensors). IMU data is used only for inter-frame integration and does not trigger additional publishes.

- **Subscriber-gated**: Every topic checks `get_subscription_count() > 0` before publishing. Topics with no subscribers are silently skipped.

- **`_corrected` topics**: Triggered by `GlobalMappingCallbacks::on_update_submaps`, i.e., only after the global mapping module completes a submap optimization or loop closure. These topics will not appear until sufficient map coverage is accumulated.

- **`~/map`**: Published at most once every **10 seconds** and only when at least one subscriber exists. It merges all current submaps into a single point cloud before publishing.

---

## Notes for Livox LiDAR

Livox sensors store per-point timestamps as **float64 absolute nanoseconds**. glim detects this automatically (`autoconf_perpoint_times: true`) and converts them to relative seconds. To suppress the related warnings, set in `config_sensors.json`:

```json
"autoconf_perpoint_times": false,
"perpoint_relative_time": false,
"perpoint_time_scale": 1e-9
```


