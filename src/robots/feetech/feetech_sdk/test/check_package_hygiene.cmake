# Package hygiene check for feetech_sdk:
# 1. No ROS runtime dependencies (rclcpp / rclpy / rosidl / message packages).
# 2. No unpinned network fetches: ExternalProject is forbidden; FetchContent
#    is allowed only when its GIT_TAG pins an exact 40-hex revision.
# 3. Public headers and sources do not include ROS headers.
#
# Build/test tooling (ament_cmake, ament_cmake_gtest, GTest) is allowed.

set(ERRORS "")

file(READ "${PKG_SOURCE_DIR}/package.xml" PACKAGE_XML)
file(READ "${PKG_SOURCE_DIR}/CMakeLists.txt" CMAKE_LISTS)

set(FORBIDDEN_DEP_PATTERNS
  "rclcpp" "rclpy" "rcl_interfaces" "rosidl" "sensor_msgs" "geometry_msgs"
  "std_msgs" "builtin_interfaces" "libstatistics_collector" "rcutils")

foreach(PATTERN IN LISTS FORBIDDEN_DEP_PATTERNS)
  # package.xml: forbidden inside <depend>/<exec_depend>/<build_depend>.
  string(REGEX MATCH "<([a-z_]*depend)>[^<]*${PATTERN}[^<]*</\\1>" DEP_MATCH "${PACKAGE_XML}")
  if(DEP_MATCH)
    list(APPEND ERRORS "package.xml declares a ROS runtime dependency: ${DEP_MATCH}")
  endif()
  # CMakeLists: forbidden in find_package / target_link_libraries tokens.
  string(REGEX MATCH "find_package\\([^)]*${PATTERN}" CMAKE_MATCH "${CMAKE_LISTS}")
  if(CMAKE_MATCH)
    list(APPEND ERRORS "CMakeLists.txt finds a forbidden package: ${CMAKE_MATCH}")
  endif()
endforeach()

string(REGEX MATCH "ExternalProject|git clone" FETCH_MATCH "${CMAKE_LISTS}")
if(FETCH_MATCH)
  list(APPEND ERRORS "CMakeLists.txt performs an unpinned fetch: ${FETCH_MATCH}")
endif()
string(REGEX MATCH "GIT_REPOSITORY[^)]*" GIT_DECL "${CMAKE_LISTS}")
if(GIT_DECL)
  string(REGEX MATCH "GIT_TAG[ \t]+[0-9a-fA-F]+" TAG_MATCH "${GIT_DECL}")
  if(TAG_MATCH)
    string(REGEX REPLACE "GIT_TAG[ \t]+([0-9a-fA-F]+)" "\\1" TAG_HASH "${TAG_MATCH}")
    string(LENGTH "${TAG_HASH}" TAG_LEN)
  endif()
  if(NOT TAG_MATCH OR NOT TAG_LEN EQUAL 40)
    list(APPEND ERRORS "FetchContent GIT_REPOSITORY must pin GIT_TAG to an exact 40-hex revision: ${GIT_DECL}")
  endif()
endif()

file(GLOB_RECURSE SDK_SOURCES
  "${PKG_SOURCE_DIR}/include/*.hpp"
  "${PKG_SOURCE_DIR}/include/*.h"
  "${PKG_SOURCE_DIR}/src/*.cpp"
  "${PKG_SOURCE_DIR}/src/*.hpp")
foreach(SOURCE_FILE IN LISTS SDK_SOURCES)
  file(READ "${SOURCE_FILE}" SOURCE_TEXT)
  string(REGEX MATCH "#include *<[rR](cl|os)" ROS_INCLUDE_MATCH "${SOURCE_TEXT}")
  if(ROS_INCLUDE_MATCH)
    list(APPEND ERRORS "${SOURCE_FILE} includes a ROS header: ${ROS_INCLUDE_MATCH}")
  endif()
endforeach()

if(ERRORS)
  foreach(ERROR_MSG IN LISTS ERRORS)
    message(FATAL_ERROR "[feetech_sdk hygiene] ${ERROR_MSG}")
  endforeach()
endif()
message(STATUS "[feetech_sdk hygiene] OK: no ROS runtime dependencies, network fetches are revision-pinned")
