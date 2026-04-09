cmake_minimum_required(VERSION 3.8)
project(global_relocalization)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── ROS2 dependencies ──────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

# ── Third-party ────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(small_gicp REQUIRED)

# ── OpenMP (optional but recommended for small_gicp) ───────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ── Include directories ────────────────────────────────────────────────
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ── Executable ────────────────────────────────────────────────────────
add_executable(global_relocalization_node
  src/global_relocalization_node.cpp
)

ament_target_dependencies(global_relocalization_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_eigen
  pcl_ros
  pcl_conversions
)

target_link_libraries(global_relocalization_node
  small_gicp::small_gicp
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(global_relocalization_node OpenMP::OpenMP_CXX)
endif()

# ── Install ────────────────────────────────────────────────────────────
install(TARGETS global_relocalization_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
