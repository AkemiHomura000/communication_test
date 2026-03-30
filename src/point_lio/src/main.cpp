// =====================================================================
// main.cpp — PointLIO 程序入口
// =====================================================================

#include "LaserMappingNode.h"

#include <csignal>
#include <rclcpp/rclcpp.hpp>

// 全局退出标志（LaserMappingNode.cpp 中通过 extern 访问）
bool g_flg_exit = false;

static void SigHandle(int /*sig*/)
{
  g_flg_exit = true;
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<LaserMappingNode>();
  node->postInit();   // shared_ptr 就绪后初始化 TF broadcaster/listener

  signal(SIGINT, SigHandle);

  rclcpp::Rate rate(500);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  while (rclcpp::ok() && !g_flg_exit)
  {
    executor.spin_some();
    node->spin_once();
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
