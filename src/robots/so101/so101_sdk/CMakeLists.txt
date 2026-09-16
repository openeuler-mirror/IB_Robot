cmake_minimum_required(VERSION 3.10)
project(so101_sdk)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)
find_package(feetech_sdk REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(${PROJECT_NAME} STATIC
  src/arm.cpp
  src/calibration.cpp
  src/leader.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

ament_target_dependencies(${PROJECT_NAME} feetech_sdk nlohmann_json)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(feetech_sdk nlohmann_json)

# ---------------------------------------------------------------------------
# Python bindings (pybind11): exposes Arm/LeaderArm/SimControl to Python.
# Required by leader acquisition; missing bindings must fail the build.
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
pybind11_add_module(so101_sdk_py python/bindings.cpp)
target_link_libraries(so101_sdk_py PRIVATE ${PROJECT_NAME})
install(TARGETS so101_sdk_py
  DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_calibration test/test_calibration.cpp)
  target_link_libraries(test_calibration ${PROJECT_NAME})

  ament_add_gtest(test_arm test/test_arm.cpp)
  target_link_libraries(test_arm ${PROJECT_NAME})

  ament_add_gtest(test_leader test/test_leader.cpp)
  target_link_libraries(test_leader ${PROJECT_NAME})
  add_test(NAME test_python_bindings
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_python_bindings.py
      $<TARGET_FILE:so101_sdk_py>)
endif()

ament_package()
