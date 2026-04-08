cmake_minimum_required(VERSION 3.8)
project(lidar_merger)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(lidar_merger_node
  src/lidar_merger_node.cpp
)

target_link_libraries(lidar_merger_node
  Eigen3::Eigen
)

ament_target_dependencies(lidar_merger_node
  rclcpp
  sensor_msgs
  message_filters
)

install(TARGETS lidar_merger_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
