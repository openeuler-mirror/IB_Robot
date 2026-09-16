cmake_minimum_required(VERSION 3.10)
project(so101_motion)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/motion_server.py
    scripts/so101_placo_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Pure-Python helper module imported by so101_placo_servo_node.py at runtime.
install(
  FILES
    scripts/so101_placo_kinematics.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # ROS launch_testing pytest plugins are incompatible with the venv pytest.
  ament_add_pytest_test(
    test_motion_server_lifecycle
    test/test_motion_server_lifecycle.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  ament_add_pytest_test(
    test_so101_placo_home
    test/test_so101_placo_home.py
    TIMEOUT 60
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  ament_add_pytest_test(
    test_ik_workers_launch
    test/test_ik_workers_launch.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
endif()

ament_package()
