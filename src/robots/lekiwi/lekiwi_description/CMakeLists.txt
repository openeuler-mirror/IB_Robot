cmake_minimum_required(VERSION 3.10)
project(lekiwi_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  mujoco
  urdf
  meshes
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_mujoco_template test/test_mujoco_template.py)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  # Ruff is this repository's Python linter (see pyproject.toml): line-length 120
  # with E501 ignored, and no pydocstyle rules selected. ament_flake8 (99 columns)
  # and ament_pep257 (D rules) therefore fail permanently on conforming code, and
  # ament_copyright enforces a header convention the repository does not follow.
  # Skipping them here keeps `colcon test` output limited to real failures
  # (the same precedent as so101_description, which also ships Python files).
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
