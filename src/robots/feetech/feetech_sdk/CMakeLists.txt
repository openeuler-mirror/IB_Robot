cmake_minimum_required(VERSION 3.10)
project(feetech_sdk)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)

# ---------------------------------------------------------------------------
# FTServo vendor sources: pinned FetchContent from the ib_robot mirror
# (same source the pre-migration so101_hardware fetched; the tag is now
# pinned to the exact revision the safe wrapper was validated against,
# instead of a moving branch). No vendor source is committed in-tree.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(ftservo
  GIT_REPOSITORY https://gitcode.com/ib_robot/FTServo_Linux.git
  GIT_TAG        06fd3356dbd7bccd886b5a70d7ae0fccc6c76d38
)
FetchContent_MakeAvailable(ftservo)
set(VENDOR_DIR ${ftservo_SOURCE_DIR}/src)
add_library(ftservo_vendored STATIC
  ${VENDOR_DIR}/SCS.cpp
  ${VENDOR_DIR}/SCSCL.cpp
  ${VENDOR_DIR}/HLSCL.cpp
  ${VENDOR_DIR}/SMS_STS.cpp
  ${VENDOR_DIR}/SCSerial.cpp
)
target_include_directories(ftservo_vendored PUBLIC
  $<BUILD_INTERFACE:${VENDOR_DIR}>)
target_compile_options(ftservo_vendored PRIVATE -Wno-vla)
set_target_properties(ftservo_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# feetech_sdk: ROS-free public library.
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC
  src/bus.cpp
  src/conversion.cpp
  src/rollback.cpp
  src/safe_sms_sts.cpp
  src/sim_sms_sts.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# Vendor headers are PRIVATE: they never appear in installed public headers,
# but internal sources and unit tests need them.
target_include_directories(${PROJECT_NAME} PRIVATE ${VENDOR_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE ftservo_vendored)

install(TARGETS ${PROJECT_NAME} ftservo_vendored
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(FILES ${ftservo_SOURCE_DIR}/LICENSE
  DESTINATION share/${PROJECT_NAME}/third_party/ftservo)

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} ftservo_vendored)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Package hygiene: no ROS runtime dependencies, no network fetches.
  add_test(
    NAME package_hygiene
    COMMAND ${CMAKE_COMMAND}
      -DPKG_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_package_hygiene.cmake)

  set(SDK_TEST_LIBS ${PROJECT_NAME})

  ament_add_gtest(test_conversion test/test_conversion.cpp)
  target_link_libraries(test_conversion ${SDK_TEST_LIBS})

  ament_add_gtest(test_rollback test/test_rollback.cpp)
  target_link_libraries(test_rollback ${SDK_TEST_LIBS})

  ament_add_gtest(test_bus_lifecycle test/test_bus_lifecycle.cpp)
  target_link_libraries(test_bus_lifecycle ${SDK_TEST_LIBS})

  ament_add_gtest(test_bus_io test/test_bus_io.cpp)
  target_link_libraries(test_bus_io ${SDK_TEST_LIBS})

  ament_add_gtest(test_config_atomicity test/test_config_atomicity.cpp)
  target_link_libraries(test_config_atomicity ${SDK_TEST_LIBS})

  ament_add_gtest(test_emergency_release test/test_emergency_release.cpp)
  target_link_libraries(test_emergency_release ${SDK_TEST_LIBS})

  ament_add_gtest(test_sim_transport test/test_sim_transport.cpp)
  target_link_libraries(test_sim_transport ${SDK_TEST_LIBS})

  ament_add_gtest(test_serial_hardening test/test_serial_hardening.cpp)
  target_include_directories(test_serial_hardening PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${VENDOR_DIR})
  target_link_libraries(test_serial_hardening ${SDK_TEST_LIBS})
endif()

ament_package()
