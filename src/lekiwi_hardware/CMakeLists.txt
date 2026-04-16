cmake_minimum_required(VERSION 3.10)
project(lekiwi_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-vla)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nlohmann_json REQUIRED)

# ===== Fetch Feetech SDK via CMake =====
include(FetchContent)
FetchContent_Declare(
  ftservo_sdk_external
  GIT_REPOSITORY https://github.com/ftservo/FTServo_Linux.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(ftservo_sdk_external)

# Build Feetech SDK from fetched source
add_library(ftservo_sdk STATIC
  ${ftservo_sdk_external_SOURCE_DIR}/src/SCS.cpp
  ${ftservo_sdk_external_SOURCE_DIR}/src/SCSCL.cpp
  ${ftservo_sdk_external_SOURCE_DIR}/src/HLSCL.cpp
  ${ftservo_sdk_external_SOURCE_DIR}/src/SMS_STS.cpp
  ${ftservo_sdk_external_SOURCE_DIR}/src/SCSerial.cpp
)
set_target_properties(ftservo_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ftservo_sdk PUBLIC
  $<BUILD_INTERFACE:${ftservo_sdk_external_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

# ===== Build hardware interface plugin =====
add_library(lekiwi_hardware_interface SHARED
  src/lekiwi_system_hardware.cpp
)
target_include_directories(lekiwi_hardware_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${ftservo_sdk_external_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lekiwi_hardware_interface ftservo_sdk)
ament_target_dependencies(lekiwi_hardware_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

# Pluginlib export
pluginlib_export_plugin_description_file(hardware_interface lekiwi_hardware_plugin.xml)

# ===== Config Files =====
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ===== Tools =====
add_executable(scan_motors tools/scan_motors.cpp)
target_link_libraries(scan_motors ftservo_sdk)
install(TARGETS scan_motors DESTINATION lib/${PROJECT_NAME})

# ===== Testing =====
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ===== Install =====
install(TARGETS lekiwi_hardware_interface ftservo_sdk
  EXPORT lekiwi_hardware_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(lekiwi_hardware_interface ftservo_sdk)
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

ament_package()
