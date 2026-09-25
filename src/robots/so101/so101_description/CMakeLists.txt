cmake_minimum_required(VERSION 3.10)
project(so101_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


# install directories
install(
  DIRECTORY launch urdf meshes rviz mujoco
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_single_arm_model test/test_single_arm_model.py)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  # Ruff is this repository's Python linter (see pyproject.toml): line-length 120
  # with E501 ignored, and no pydocstyle rules selected. ament_flake8 (99 columns)
  # and ament_pep257 (D rules) therefore fail permanently on conforming code, and
  # ament_copyright enforces a header convention the repository does not follow.
  # Skipping them here keeps `colcon test` output limited to real failures.
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
