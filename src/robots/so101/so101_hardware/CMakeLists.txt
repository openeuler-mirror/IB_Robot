cmake_minimum_required(VERSION 3.10)
project(so101_hardware)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-vla)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(ibrobot_msgs REQUIRED)
find_package(feetech_sdk REQUIRED)
find_package(so101_sdk REQUIRED)
find_package(nlohmann_json REQUIRED)

# ===== Hardware interface plugin (thin adapter over so101_sdk) =====
add_library(so101_hardware_interface SHARED
  src/so101_system_hardware.cpp
)
target_include_directories(so101_hardware_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(so101_hardware_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  ibrobot_msgs
  feetech_sdk
  so101_sdk
  nlohmann_json
)

# Pluginlib export
pluginlib_export_plugin_description_file(hardware_interface so101_hardware_plugin.xml)

# ===== Python Package =====
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts without .py suffix for ROS 2 standards
install(PROGRAMS scripts/calibrate_arm.py DESTINATION lib/${PROJECT_NAME} RENAME calibrate_arm)
install(PROGRAMS scripts/arm_calibration_transfer.py DESTINATION lib/${PROJECT_NAME} RENAME arm_calibration_transfer)
install(PROGRAMS scripts/arm_calibration_checker.py DESTINATION lib/${PROJECT_NAME} RENAME arm_calibration_checker)
install(PROGRAMS scripts/leader_arm_pub.py DESTINATION lib/${PROJECT_NAME} RENAME leader_arm_pub)
install(PROGRAMS scripts/motor_test.py DESTINATION lib/${PROJECT_NAME} RENAME motor_test)
install(PROGRAMS scripts/motor_bridge.py DESTINATION lib/${PROJECT_NAME} RENAME motor_bridge)
install(PROGRAMS scripts/read_motor_steps.py DESTINATION lib/${PROJECT_NAME} RENAME read_motor_steps)

# ===== Launch Files =====
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# ===== Config Files =====
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ===== Testing =====
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_so101_hardware test/test_so101_hardware.cpp)
  target_link_libraries(test_so101_hardware so101_hardware_interface)

  ament_add_pytest_test(test_calibration_py test/test_calibration.py)
  ament_add_pytest_test(test_transfer_py test/test_transfer.py)
  ament_add_pytest_test(test_checker_py test/test_checker.py)

  ament_lint_auto_find_test_dependencies()
endif()

# ===== Install =====
install(TARGETS so101_hardware_interface
  EXPORT so101_hardware_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(so101_hardware_interface)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle ibrobot_msgs feetech_sdk so101_sdk)
ament_package()
