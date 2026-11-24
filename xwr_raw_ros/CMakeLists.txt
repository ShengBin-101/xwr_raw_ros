# Modified by Chan Sheng Bin, 2026: rewritten for ROS 2 (ament_cmake). See NOTICE.
cmake_minimum_required(VERSION 3.8)
project(xwr_raw_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadarFrame.msg"
  "msg/RadarFrameStamped.msg"
  "msg/RadarFrameFull.msg"
  DEPENDENCIES std_msgs
)

# The DSP and capture code, importable as `xwr_raw`
ament_python_install_package(xwr_raw PACKAGE_DIR src/xwr_raw)

install(PROGRAMS nodes/radar.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch configs udev DESTINATION share/${PROJECT_NAME})

ament_package()
