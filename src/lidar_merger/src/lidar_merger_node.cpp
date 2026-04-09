#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <livox_ros_driver2/msg/custom_point.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

class LidarMergerNode : public rclcpp::Node
{
public:
  using PointCloud2  = sensor_msgs::msg::PointCloud2;
  using CustomMsg    = livox_ros_driver2::msg::CustomMsg;
  using CustomPoint  = livox_ros_driver2::msg::CustomPoint;

  using SyncPolicyPC2    = message_filters::sync_policies::ApproximateTime<PointCloud2, PointCloud2>;
  using SyncPC2          = message_filters::Synchronizer<SyncPolicyPC2>;
  using SyncPolicyCustom = message_filters::sync_policies::ApproximateTime<CustomMsg, CustomMsg>;
  using SyncCustom       = message_filters::Synchronizer<SyncPolicyCustom>;

  LidarMergerNode() : Node("lidar_merger")
  {
    // ── Parameters ──────────────────────────────────────────────────────────
    // Point-cloud message format: "pointcloud2" or "custom_msg"
    declare_parameter("cloud_format",    "pointcloud2");
    declare_parameter("primary_topic",   "/livox/lidar_192_168_1_183");
    declare_parameter("secondary_topic", "/livox/lidar_192_168_1_133");
    declare_parameter("output_topic",    "/livox/lidar_merged");
    // Extrinsic: secondary → primary frame  (RPY in degrees, translation in metres)
    declare_parameter("extrinsic.roll",  0.0);
    declare_parameter("extrinsic.pitch", 0.0);
    declare_parameter("extrinsic.yaw",   0.0);
    declare_parameter("extrinsic.x",     0.0);
    declare_parameter("extrinsic.y",     0.0);
    declare_parameter("extrinsic.z",     0.0);
    // Synchronisation
    declare_parameter("sync_queue_size", 10);
    declare_parameter("max_delay",       0.1);  // seconds
    // Per-point timestamp alignment:
    //   true  → shift secondary per-point timestamps by the header time delta
    //           (use when the two LiDARs are NOT hardware-synced)
    //   false → leave per-point timestamps untouched
    //           (use when both LiDARs share a common PTP/GPS time source)
    declare_parameter("sync_point_timestamps", true);
    // How often (seconds) to print the periodic statistics summary (0 = disable)
    declare_parameter("print_interval", 5.0);
    // Clip secondary cloud to the primary frame's time window
    //   [t_primary_header, t_primary_header + 1/lidar_freq]
    // Prevents secondary points beyond this window from corrupting
    // per-point motion compensation in the downstream SLAM algorithm.
    // Only effective when sync_point_timestamps=true and timestamp field exists.
    declare_parameter("clip_secondary", true);
    declare_parameter("lidar_freq",     10.0);  // Hz

    const auto primary_topic   = get_parameter("primary_topic").as_string();
    const auto secondary_topic = get_parameter("secondary_topic").as_string();
    const auto output_topic    = get_parameter("output_topic").as_string();

    const double deg2rad = M_PI / 180.0;
    const double roll    = get_parameter("extrinsic.roll").as_double()  * deg2rad;
    const double pitch   = get_parameter("extrinsic.pitch").as_double() * deg2rad;
    const double yaw     = get_parameter("extrinsic.yaw").as_double()   * deg2rad;
    const double tx      = get_parameter("extrinsic.x").as_double();
    const double ty      = get_parameter("extrinsic.y").as_double();
    const double tz      = get_parameter("extrinsic.z").as_double();

    const int    sync_q  = get_parameter("sync_queue_size").as_int();
    const double maxd    = get_parameter("max_delay").as_double();
    sync_point_timestamps_ = get_parameter("sync_point_timestamps").as_bool();
    print_interval_        = get_parameter("print_interval").as_double();
    clip_secondary_        = get_parameter("clip_secondary").as_bool();
    frame_period_          = 1.0 / get_parameter("lidar_freq").as_double();
    cloud_format_          = get_parameter("cloud_format").as_string();

    // ── Pre-compute 4×4 transform (secondary → primary) ────────────────────
    // Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)   (extrinsic ZYX)
    build_transform(roll, pitch, yaw, tx, ty, tz);

    RCLCPP_INFO(get_logger(),
      "[lidar_merger] format=%s  extrinsic RPY=(%.3f, %.3f, %.3f) deg  T=(%.4f, %.4f, %.4f) m",
      cloud_format_.c_str(),
      roll / deg2rad, pitch / deg2rad, yaw / deg2rad, tx, ty, tz);

    // ── Publisher & subscribers ───────────────────────────────────────────
    if (cloud_format_ == "custom_msg") {
      setup_custom_msg(primary_topic, secondary_topic, output_topic, sync_q, maxd);
    } else {
      setup_pointcloud2(primary_topic, secondary_topic, output_topic, sync_q, maxd);
    }

    RCLCPP_INFO(get_logger(),
      "[lidar_merger] ready  primary=%s  secondary=%s  out=%s"
      "  sync_point_ts=%s  clip=%s(%.1fHz)  print_interval=%.1fs",
      primary_topic.c_str(), secondary_topic.c_str(), output_topic.c_str(),
      sync_point_timestamps_ ? "true" : "false",
      clip_secondary_ ? "true" : "false",
      1.0 / frame_period_,
      print_interval_);
  }

private:
  // ── Setup helpers ────────────────────────────────────────────────────────
  void setup_pointcloud2(const std::string & pri, const std::string & sec,
                         const std::string & out, int sync_q, double maxd)
  {
    pub_ = create_publisher<PointCloud2>(out, rclcpp::SensorDataQoS());
    primary_sub_.subscribe(this, pri,   rmw_qos_profile_sensor_data);
    secondary_sub_.subscribe(this, sec, rmw_qos_profile_sensor_data);
    sync_pc2_ = std::make_shared<SyncPC2>(SyncPolicyPC2(sync_q), primary_sub_, secondary_sub_);
    sync_pc2_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(maxd));
    sync_pc2_->registerCallback(&LidarMergerNode::sync_cb_pc2, this);
  }

  void setup_custom_msg(const std::string & pri, const std::string & sec,
                        const std::string & out, int sync_q, double maxd)
  {
    pub_custom_ = create_publisher<CustomMsg>(out, rclcpp::SensorDataQoS());
    primary_custom_sub_.subscribe(this, pri,   rmw_qos_profile_sensor_data);
    secondary_custom_sub_.subscribe(this, sec, rmw_qos_profile_sensor_data);
    sync_custom_ = std::make_shared<SyncCustom>(
      SyncPolicyCustom(sync_q), primary_custom_sub_, secondary_custom_sub_);
    sync_custom_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(maxd));
    sync_custom_->registerCallback(&LidarMergerNode::sync_cb_custom, this);
  }
  // ── Build 4×4 homogeneous transform ─────────────────────────────────────
  void build_transform(double roll, double pitch, double yaw,
                       double tx,   double ty,    double tz)
  {
    Eigen::AngleAxisf Rx(static_cast<float>(roll),  Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf Ry(static_cast<float>(pitch), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf Rz(static_cast<float>(yaw),   Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f R = (Rz * Ry * Rx).matrix();

    T_ = Eigen::Matrix4f::Identity();
    T_.block<3, 3>(0, 0) = R;
    T_(0, 3) = static_cast<float>(tx);
    T_(1, 3) = static_cast<float>(ty);
    T_(2, 3) = static_cast<float>(tz);
  }

  // ── Synchronised callback (PointCloud2) ──────────────────────────────────
  void sync_cb_pc2(const PointCloud2::ConstSharedPtr & primary,
                   const PointCloud2::ConstSharedPtr & secondary)
  {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    // ── Timestamp information ─────────────────────────────────────────────
    const double t_primary   = rclcpp::Time(primary->header.stamp).seconds();
    const double t_secondary = rclcpp::Time(secondary->header.stamp).seconds();
    const double header_gap  = t_primary - t_secondary;          // seconds
    const double now_wall    = rclcpp::Clock().now().seconds();
    const double latency_pri = now_wall - t_primary;             // receive latency
    const double latency_sec = now_wall - t_secondary;

    // Per-point timestamp shift
    const double ts_delta = sync_point_timestamps_ ? header_gap : 0.0;

    // ── Print per-frame timestamp diagnostics ─────────────────────────────
    RCLCPP_INFO(get_logger(),
      "[ts] primary=%.6f  secondary=%.6f  header_gap=%+.4fms  "
      "latency_pri=%.1fms  latency_sec=%.1fms",
      t_primary, t_secondary,
      header_gap  * 1e3,
      latency_pri * 1e3,
      latency_sec * 1e3);

    // Transform secondary cloud into primary frame; adjust per-point timestamps
    PointCloud2 sec_transformed;
    transform_cloud(*secondary, sec_transformed, ts_delta);

    // Clip secondary points to the primary frame's time window
    // Window: [t_primary, t_primary + frame_period]
    // After ts adjustment the secondary per-point timestamps are in the
    // primary time domain, so this comparison is valid.
    uint32_t clipped_pts = 0;
    if (clip_secondary_ && sync_point_timestamps_ && ts_off_ >= 0) {
      clip_cloud(sec_transformed, t_primary, t_primary + frame_period_, clipped_pts);
    }

    // Merge primary + transformed secondary
    PointCloud2 merged;
    merge_clouds(*primary, sec_transformed, merged);

    // Output header (timestamp + frame_id) from the primary cloud
    merged.header = primary->header;

    pub_->publish(std::move(merged));

    RCLCPP_INFO(get_logger(),
      "[clip] sec_pts_in=%u  clipped=%u  sec_pts_kept=%u",
      secondary->width * secondary->height,
      clipped_pts,
      sec_transformed.width * sec_transformed.height);

    // ── Performance accounting ────────────────────────────────────────────
    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    ++stats_.frames;
    stats_.total_pts      += merged.width * merged.height;
    stats_.total_clipped  += clipped_pts;
    stats_.sum_ms          = stats_.sum_ms + elapsed_ms;
    stats_.min_ms          = std::min(stats_.min_ms, elapsed_ms);
    stats_.max_ms          = std::max(stats_.max_ms, elapsed_ms);
    stats_.sum_gap_ms     += std::abs(header_gap) * 1e3;
    stats_.sum_lat_ms     += latency_pri * 1e3;

    maybe_print_stats(t_primary);
  }

  // ── Synchronised callback (CustomMsg) ────────────────────────────────────
  void sync_cb_custom(const CustomMsg::ConstSharedPtr & primary,
                      const CustomMsg::ConstSharedPtr & secondary)
  {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const int64_t t_pri_ns  = static_cast<int64_t>(primary->timebase);
    const int64_t t_sec_ns  = static_cast<int64_t>(secondary->timebase);
    const double  t_primary   = static_cast<double>(t_pri_ns) * 1e-9;
    const double  t_secondary = static_cast<double>(t_sec_ns) * 1e-9;
    const double  header_gap  = t_primary - t_secondary;
    const double  now_wall    = rclcpp::Clock().now().seconds();

    RCLCPP_INFO(get_logger(),
      "[ts] primary=%.6f  secondary=%.6f  header_gap=%+.4fms  "
      "latency_pri=%.1fms  latency_sec=%.1fms",
      t_primary, t_secondary,
      header_gap  * 1e3,
      (now_wall - t_primary)   * 1e3,
      (now_wall - t_secondary) * 1e3);

    // Transform secondary points (xyz) and re-express offsets relative to
    // the primary's timebase.  Always performed so the merged message has a
    // single consistent timebase.
    CustomMsg sec_transformed;
    transform_custom(*secondary, sec_transformed, t_pri_ns, t_sec_ns);

    // Optionally clip secondary points to the primary frame's time window.
    // After adjustment sec offsets are in [0, frame_period_ns] for in-window pts.
    uint32_t clipped_pts = 0;
    if (clip_secondary_ && sync_point_timestamps_) {
      const uint64_t win_end_ns = static_cast<uint64_t>(frame_period_ * 1e9);
      clip_custom(sec_transformed, 0ULL, win_end_ns, clipped_pts);
    }

    // Merge: primary points first, then transformed secondary points
    CustomMsg merged = *primary;
    merged.points.insert(merged.points.end(),
                         sec_transformed.points.begin(),
                         sec_transformed.points.end());
    merged.point_num = static_cast<uint32_t>(merged.points.size());

    pub_custom_->publish(std::move(merged));

    RCLCPP_INFO(get_logger(),
      "[clip] sec_pts_in=%u  clipped=%u  sec_pts_kept=%zu",
      secondary->point_num, clipped_pts, sec_transformed.points.size());

    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    ++stats_.frames;
    stats_.total_pts     += merged.point_num;
    stats_.total_clipped += clipped_pts;
    stats_.sum_ms         = stats_.sum_ms + elapsed_ms;
    stats_.min_ms         = std::min(stats_.min_ms, elapsed_ms);
    stats_.max_ms         = std::max(stats_.max_ms, elapsed_ms);
    stats_.sum_gap_ms    += std::abs(header_gap) * 1e3;
    stats_.sum_lat_ms    += (now_wall - t_primary) * 1e3;

    maybe_print_stats(t_primary);
  }

  // ── Periodic statistics summary ──────────────────────────────────────────
  void maybe_print_stats(double now_sec)
  {
    if (print_interval_ <= 0.0) return;
    if (stats_.next_print < 0.0) stats_.next_print = now_sec + print_interval_;
    if (now_sec < stats_.next_print) return;

    const double avg_ms     = stats_.sum_ms    / stats_.frames;
    const double avg_gap_ms = stats_.sum_gap_ms / stats_.frames;
    const double avg_lat_ms = stats_.sum_lat_ms / stats_.frames;
    const double avg_pts    = static_cast<double>(stats_.total_pts) / stats_.frames;

    const double avg_clipped = static_cast<double>(stats_.total_clipped) / stats_.frames;

    RCLCPP_INFO(get_logger(),
      "[stats/%ds] frames=%u  pts/frame=%.0f  clipped/frame=%.0f  "
      "cb_time: avg=%.2f min=%.2f max=%.2f ms  "
      "header_gap(abs): avg=%.2fms  latency_pri: avg=%.1fms",
      static_cast<int>(print_interval_),
      stats_.frames, avg_pts, avg_clipped,
      avg_ms, stats_.min_ms, stats_.max_ms,
      avg_gap_ms, avg_lat_ms);

    stats_ = Stats{};
    stats_.next_print = now_sec + print_interval_;
  }

  // ── Apply pre-computed transform to x/y/z, optionally shift per-point ts ─
  // ts_delta: value added to every point's "timestamp" field (seconds).
  //           Pass 0.0 to leave timestamps unchanged.
  void transform_cloud(const PointCloud2 & in, PointCloud2 & out, double ts_delta)
  {
    out = in;  // deep-copy: preserves header, fields, all raw bytes

    // Resolve field offsets once (lazily cached)
    if (!offsets_resolved_) {
      for (const auto & f : in.fields) {
        if      (f.name == "x")         x_off_  = static_cast<int>(f.offset);
        else if (f.name == "y")         y_off_  = static_cast<int>(f.offset);
        else if (f.name == "z")         z_off_  = static_cast<int>(f.offset);
        else if (f.name == "timestamp") ts_off_ = static_cast<int>(f.offset);
      }
      if (x_off_ < 0 || y_off_ < 0 || z_off_ < 0) {
        RCLCPP_ERROR_ONCE(get_logger(), "Secondary cloud has no x/y/z fields – skipping transform");
        return;
      }
      if (ts_off_ < 0) {
        RCLCPP_WARN_ONCE(get_logger(),
          "Secondary cloud has no \"timestamp\" field – per-point timestamp sync skipped");
      }
      offsets_resolved_ = true;
    }

    const uint32_t point_step = in.point_step;
    const uint32_t n_pts      = in.width * in.height;
    uint8_t *      data       = out.data.data();

    // Cache rotation+translation matrix elements for the hot loop
    const float m00 = T_(0,0), m01 = T_(0,1), m02 = T_(0,2), m03 = T_(0,3);
    const float m10 = T_(1,0), m11 = T_(1,1), m12 = T_(1,2), m13 = T_(1,3);
    const float m20 = T_(2,0), m21 = T_(2,1), m22 = T_(2,2), m23 = T_(2,3);

    const bool adjust_ts = (sync_point_timestamps_ && ts_off_ >= 0 && ts_delta != 0.0);

    for (uint32_t i = 0; i < n_pts; ++i) {
      uint8_t * p = data + i * point_step;

      // ── Transform xyz ──────────────────────────────────────────────────
      float x, y, z;
      std::memcpy(&x, p + x_off_, sizeof(float));
      std::memcpy(&y, p + y_off_, sizeof(float));
      std::memcpy(&z, p + z_off_, sizeof(float));

      const float nx = m00*x + m01*y + m02*z + m03;
      const float ny = m10*x + m11*y + m12*z + m13;
      const float nz = m20*x + m21*y + m22*z + m23;

      std::memcpy(p + x_off_, &nx, sizeof(float));
      std::memcpy(p + y_off_, &ny, sizeof(float));
      std::memcpy(p + z_off_, &nz, sizeof(float));

      // ── Shift per-point timestamp (float64, seconds) ───────────────────
      if (adjust_ts) {
        double ts;
        std::memcpy(&ts, p + ts_off_, sizeof(double));
        ts += ts_delta;
        std::memcpy(p + ts_off_, &ts, sizeof(double));
      }
    }
  }

  // ── Clip secondary cloud to time window [t_min, t_max] ─────────────────
  // Modifies `cloud` in-place. Points outside the window are compacted out.
  // `clipped_out` receives the number of removed points.
  void clip_cloud(PointCloud2 & cloud, double t_min, double t_max,
                  uint32_t & clipped_out)
  {
    clipped_out = 0;
    if (ts_off_ < 0) return;

    const uint32_t point_step = cloud.point_step;
    const uint32_t n_in       = cloud.width * cloud.height;
    uint8_t *      src        = cloud.data.data();
    uint32_t       n_out      = 0;

    for (uint32_t i = 0; i < n_in; ++i) {
      const uint8_t * p = src + i * point_step;
      double ts;
      std::memcpy(&ts, p + ts_off_, sizeof(double));
      if (ts >= t_min && ts <= t_max) {
        if (n_out != i) {
          // Compact kept point to the front of the buffer
          std::memmove(src + n_out * point_step, p, point_step);
        }
        ++n_out;
      }
    }

    clipped_out   = n_in - n_out;
    cloud.width   = n_out;
    cloud.height  = 1;
    cloud.row_step = n_out * point_step;
    cloud.data.resize(cloud.row_step);
  }

  // ── Concatenate two PointCloud2 buffers ──────────────────────────────────
  // Assumes both clouds share the same point_step / field layout (guaranteed
  // for two MID360s driven with identical livox_ros_driver2 settings).
  void merge_clouds(const PointCloud2 & c1,
                    const PointCloud2 & c2,
                    PointCloud2       & out)
  {
    if (c1.point_step != c2.point_step) {
      RCLCPP_WARN_ONCE(get_logger(),
        "point_step mismatch (%u vs %u) – publishing primary cloud only",
        c1.point_step, c2.point_step);
      out = c1;
      return;
    }

    const uint32_t n1 = c1.width * c1.height;
    const uint32_t n2 = c2.width * c2.height;

    out             = c1;            // copy fields, point_step, is_bigendian …
    out.height      = 1;
    out.width       = n1 + n2;
    out.row_step    = out.width * c1.point_step;
    out.is_dense    = false;
    out.data.resize(out.row_step);

    std::memcpy(out.data.data(),                  c1.data.data(), c1.data.size());
    std::memcpy(out.data.data() + c1.data.size(), c2.data.data(), c2.data.size());
  }

  // ── Transform CustomMsg secondary and re-express offsets vs primary base ─
  // t_pri_ns / t_sec_ns: epoch timestamps in nanoseconds (from CustomMsg::timebase).
  // The output cloud uses t_pri_ns as its timebase.
  // If sync_point_timestamps_=true, per-point offsets are shifted by (t_sec_ns - t_pri_ns)
  // so they are expressed relative to the primary's timebase.
  // If false, offsets are left unchanged (hardware-synced case; timebases should match).
  void transform_custom(const CustomMsg & in, CustomMsg & out,
                        int64_t t_pri_ns, int64_t t_sec_ns)
  {
    out = in;
    out.timebase = static_cast<uint64_t>(t_pri_ns);

    const float m00 = T_(0,0), m01 = T_(0,1), m02 = T_(0,2), m03 = T_(0,3);
    const float m10 = T_(1,0), m11 = T_(1,1), m12 = T_(1,2), m13 = T_(1,3);
    const float m20 = T_(2,0), m21 = T_(2,1), m22 = T_(2,2), m23 = T_(2,3);

    // delta: add to secondary offset_time to get value relative to primary timebase
    const int64_t delta_ns = t_sec_ns - t_pri_ns;

    for (auto & pt : out.points) {
      // ── Transform xyz ──────────────────────────────────────────────────
      const float x = pt.x, y = pt.y, z = pt.z;
      pt.x = m00*x + m01*y + m02*z + m03;
      pt.y = m10*x + m11*y + m12*z + m13;
      pt.z = m20*x + m21*y + m22*z + m23;

      // ── Adjust per-point offset ────────────────────────────────────────
      if (sync_point_timestamps_) {
        const int64_t new_off = static_cast<int64_t>(pt.offset_time) + delta_ns;
        // Clamp to uint32 range; out-of-window points are removed by clip_custom
        pt.offset_time = static_cast<uint32_t>(
          std::max(int64_t{0}, std::min(new_off, int64_t{0xFFFFFFFF})));
      }
    }
    out.point_num = static_cast<uint32_t>(out.points.size());
  }

  // ── Clip CustomMsg cloud to offset window [off_min, off_max] (ns) ────────
  // Removes points in-place whose offset_time is outside the window.
  void clip_custom(CustomMsg & cloud, uint64_t off_min, uint64_t off_max,
                   uint32_t & clipped_out)
  {
    auto it = std::remove_if(cloud.points.begin(), cloud.points.end(),
      [off_min, off_max](const CustomPoint & pt) {
        const uint64_t off = static_cast<uint64_t>(pt.offset_time);
        return off < off_min || off > off_max;
      });
    clipped_out = static_cast<uint32_t>(
      std::distance(it, cloud.points.end()));
    cloud.points.erase(it, cloud.points.end());
    cloud.point_num = static_cast<uint32_t>(cloud.points.size());
  }

  // ── Members ──────────────────────────────────────────────────────────────
  std::string cloud_format_{"pointcloud2"};

  // ── PointCloud2 mode ─────────────────────────────────────────────────────
  message_filters::Subscriber<PointCloud2> primary_sub_;
  message_filters::Subscriber<PointCloud2> secondary_sub_;
  std::shared_ptr<SyncPC2>                 sync_pc2_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;

  // ── CustomMsg mode ────────────────────────────────────────────────────────
  message_filters::Subscriber<CustomMsg>   primary_custom_sub_;
  message_filters::Subscriber<CustomMsg>   secondary_custom_sub_;
  std::shared_ptr<SyncCustom>              sync_custom_;
  rclcpp::Publisher<CustomMsg>::SharedPtr  pub_custom_;

  Eigen::Matrix4f T_{Eigen::Matrix4f::Identity()};
  bool   sync_point_timestamps_{true};
  bool   clip_secondary_{true};
  double frame_period_{0.1};   // seconds (1 / lidar_freq)
  double print_interval_{5.0};

  // Field offsets resolved lazily on first callback
  bool offsets_resolved_{false};
  int  x_off_{-1}, y_off_{-1}, z_off_{-1}, ts_off_{-1};

  // ── Performance statistics (reset each print_interval) ──────────────────
  struct Stats {
    uint32_t frames{0};
    uint64_t total_pts{0};
    uint64_t total_clipped{0};
    double   sum_ms{0.0};
    double   min_ms{std::numeric_limits<double>::max()};
    double   max_ms{0.0};
    double   sum_gap_ms{0.0};   // |header_gap|
    double   sum_lat_ms{0.0};   // receive latency (primary)
    double   next_print{-1.0};
  } stats_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarMergerNode>());
  rclcpp::shutdown();
  return 0;
}
