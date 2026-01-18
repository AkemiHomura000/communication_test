
#include "sentry_communicator/can_bus.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv)
{

  rclcpp::init(argc, argv);
  auto node = std::make_shared<sentry_communicator::CanBus>("can0", 20);
 
}
