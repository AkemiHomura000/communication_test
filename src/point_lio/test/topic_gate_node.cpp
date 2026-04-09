// =============================================================================
//  topic_gate_node.cpp
//
//  Subscribes to:
//    /livox/lidar_192_168_1_133   (livox_ros_driver2/msg/CustomMsg)
//    /livox/lidar_192_168_1_183   (livox_ros_driver2/msg/CustomMsg)
//    /livox/imu_192_168_1_183     (sensor_msgs/msg/Imu)
//
//  Re-publishes each on a mirrored topic (same name + "_relay" suffix by
//  default, or configurable via ROS params).
//
//  A separate stdin-reader thread listens for single keypresses:
//    SPACE  – toggle all topics (pause / resume)
//    1      – toggle lidar_133  only
//    2      – toggle lidar_183  only
//    3      – toggle imu_183    only
//    q / Q  – quit
//
//  The node prints the current gate state to stdout whenever it changes.
// =============================================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <termios.h>
#include <thread>
#include <unistd.h>

// ── terminal raw-mode helpers ──────────────────────────────────────────────
static struct termios g_orig_termios;

static void enableRawMode()
{
  tcgetattr(STDIN_FILENO, &g_orig_termios);
  struct termios raw = g_orig_termios;
  raw.c_lflag &= ~(ICANON | ECHO);   // no line-buffering, no echo
  raw.c_cc[VMIN]  = 0;               // non-blocking read
  raw.c_cc[VTIME] = 1;               // 0.1 s timeout
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

static void disableRawMode()
{
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
}

// ── global exit flag ───────────────────────────────────────────────────────
static std::atomic<bool> g_exit{false};
static void sigHandler(int) { g_exit = true; }

// =============================================================================
class TopicGateNode : public rclcpp::Node
{
public:
  TopicGateNode() : Node("topic_gate_node")
  {
    // ── gate flags (all start open / forwarding) ──────────────────────────
    gate_lidar133_.store(true);
    gate_lidar183_.store(true);
    gate_imu183_.store(true);

    // ── QoS: BEST_EFFORT to match Livox driver ────────────────────────────
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

    // ── publishers ────────────────────────────────────────────────────────
    pub_lidar133_ = create_publisher<livox_ros_driver2::msg::CustomMsg>(
        "/livox/lidar_192_168_1_133_relay", qos);
    pub_lidar183_ = create_publisher<livox_ros_driver2::msg::CustomMsg>(
        "/livox/lidar_192_168_1_183_relay", qos);
    pub_imu183_ = create_publisher<sensor_msgs::msg::Imu>(
        "/livox/imu_192_168_1_183_relay", qos);

    // ── subscribers ───────────────────────────────────────────────────────
    sub_lidar133_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
        "/livox/lidar_192_168_1_133", qos,
        [this](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
          if (gate_lidar133_) pub_lidar133_->publish(*msg);
        });

    sub_lidar183_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
        "/livox/lidar_192_168_1_183", qos,
        [this](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
          if (gate_lidar183_) pub_lidar183_->publish(*msg);
        });

    sub_imu183_ = create_subscription<sensor_msgs::msg::Imu>(
        "/livox/imu_192_168_1_183", qos,
        [this](sensor_msgs::msg::Imu::SharedPtr msg) {
          if (gate_imu183_) pub_imu183_->publish(*msg);
        });

    printHelp();
    printStatus();
  }

  // ── called from stdin thread ─────────────────────────────────────────────
  void handleKey(char c)
  {
    bool changed = false;
    if (c == ' ')
    {
      // all-toggle: if any is open → close all; if all closed → open all
      bool any_open = gate_lidar133_ || gate_lidar183_ || gate_imu183_;
      gate_lidar133_.store(!any_open);
      gate_lidar183_.store(!any_open);
      gate_imu183_.store(!any_open);
      changed = true;
    }
    else if (c == '1') { gate_lidar133_.store(!gate_lidar133_.load()); changed = true; }
    else if (c == '2') { gate_lidar183_.store(!gate_lidar183_.load()); changed = true; }
    else if (c == '3') { gate_imu183_.store(!gate_imu183_.load());     changed = true; }

    if (changed)
      printStatus();
  }

private:
  // ── gate flags ────────────────────────────────────────────────────────────
  std::atomic<bool> gate_lidar133_;
  std::atomic<bool> gate_lidar183_;
  std::atomic<bool> gate_imu183_;

  // ── pubs / subs ───────────────────────────────────────────────────────────
  rclcpp::Publisher<livox_ros_driver2::msg::CustomMsg>::SharedPtr pub_lidar133_;
  rclcpp::Publisher<livox_ros_driver2::msg::CustomMsg>::SharedPtr pub_lidar183_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr             pub_imu183_;

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_lidar133_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_lidar183_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr             sub_imu183_;

  // ── helpers ───────────────────────────────────────────────────────────────
  static void printHelp()
  {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║            Topic Gate Node  —  Controls              ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  SPACE  toggle ALL topics (pause / resume)           ║\n");
    printf("║  1      toggle lidar_192_168_1_133                   ║\n");
    printf("║  2      toggle lidar_192_168_1_183                   ║\n");
    printf("║  3      toggle imu_192_168_1_183                     ║\n");
    printf("║  q/Q    quit                                         ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    fflush(stdout);
  }

  void printStatus() const
  {
    auto flag = [](bool v) { return v ? "\033[32mON \033[0m" : "\033[31mOFF\033[0m"; };
    printf("[Gate] lidar_133=%s  lidar_183=%s  imu_183=%s\n",
           flag(gate_lidar133_.load()),
           flag(gate_lidar183_.load()),
           flag(gate_imu183_.load()));
    fflush(stdout);
  }
};

// =============================================================================
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  signal(SIGINT, sigHandler);

  auto node = std::make_shared<TopicGateNode>();

  // ── stdin reader thread ──────────────────────────────────────────────────
  enableRawMode();
  std::thread stdin_thread([&node]() {
    while (!g_exit && rclcpp::ok())
    {
      char c = '\0';
      if (read(STDIN_FILENO, &c, 1) == 1)
      {
        if (c == 'q' || c == 'Q')
        {
          g_exit = true;
          break;
        }
        node->handleKey(c);
      }
    }
  });

  // ── spin ─────────────────────────────────────────────────────────────────
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  while (rclcpp::ok() && !g_exit)
    executor.spin_some(std::chrono::milliseconds(10));

  g_exit = true;
  disableRawMode();
  if (stdin_thread.joinable())
    stdin_thread.join();

  rclcpp::shutdown();
  return 0;
}
