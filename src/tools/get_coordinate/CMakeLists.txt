cmake_minimum_required(VERSION 3.8)
project(get_coordinate)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(robot_msg REQUIRED)

# Use system Qt5 (no manual CMAKE_PREFIX_PATH)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include)

add_executable(get_coordinate
  src/main.cpp
  src/widget.cpp
  include/widget.h
  src/widget.ui
)

ament_target_dependencies(get_coordinate
  rclcpp
  std_msgs
  robot_msg
)

target_link_libraries(get_coordinate
  Qt5::Widgets
  yaml-cpp
)

install(TARGETS get_coordinate
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
