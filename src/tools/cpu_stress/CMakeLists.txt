cmake_minimum_required(VERSION 3.8)
project(cpu_stress)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

add_executable(cpu_stress src/cpu_stress.cpp)

target_compile_features(cpu_stress PUBLIC cxx_std_17)

target_link_libraries(cpu_stress
  pthread
  m
)

# 优化等级：Release 模式下用 O2，避免编译器把压力循环整体消除
target_compile_options(cpu_stress PRIVATE
  $<$<CONFIG:Release>:-O2>
  $<$<CONFIG:Debug>:-O0 -g>
)

install(TARGETS cpu_stress
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
