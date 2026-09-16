// Copyright 2026 IB_Robot Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Thin-adapter tests for SO101SystemHardware.
//
// Coverage relocation note (was: this file tested detail:: helpers that
// lived in the plugin header):
// - detail::disable_torque_on_abort / rollback_activation
//   -> feetech_sdk test_rollback.cpp
// - SafeSMSSTS serial read hardening
//   -> feetech_sdk test_serial_hardening.cpp
// - detail::perform_initial_sync_feedback (initial feedback sync semantics)
//   -> so101_sdk test_arm.cpp (ActivationAbortLeavesSafeState,
//      ActivationWithResetPositionsSeedsConfiguredCommands)
//
// What remains here: the adapter's HardwareInfo parsing (on_init) — the
// mapping from ros2_control's URDF hardware parameters to the SDK
// configuration.

#include <gtest/gtest.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hardware_interface/hardware_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "so101_hardware/so101_system_hardware.hpp"

namespace so101_hardware
{
struct SO101HardwareTestAccess
{
  static feetech::Bus & bus(SO101SystemHardware & hw) {return *hw.bus_;}
  static so101::Arm & arm(SO101SystemHardware & hw) {return *hw.arm_;}
  static const std::vector<double> & currents(const SO101SystemHardware & hw)
  {return hw.hw_currents_;}
  static std::chrono::steady_clock::duration write_budget(const SO101SystemHardware & hw)
  {return hw.write_retry_budget_;}
  static bool attach_bus(SO101SystemHardware & hw, std::unique_ptr<feetech::Bus> bus)
  {
    hw.bus_ = std::move(bus);
    return hw.arm_->attach_shared_bus(*hw.bus_, so101::Calibration::load(
      hw.arm_config_.calibration_file, hw.arm_config_.joint_order, hw.arm_config_.motor_ids));
  }
};
}  // namespace so101_hardware

namespace
{

hardware_interface::HardwareInfo make_arm_info(
  const std::unordered_map<std::string, std::string> & params = {},
  const std::vector<std::string> & joint_names = {"1", "2", "3", "4", "5", "6"})
{
  hardware_interface::HardwareInfo info;
  info.name = "SO101System";
  info.type = "system";
  for (const auto & [key, value] : params) {
    info.hardware_parameters[key] = value;
  }
  for (const std::string & name : joint_names) {
    hardware_interface::ComponentInfo joint;
    joint.name = name;
    joint.parameters["id"] = name;  // SO-101: joint name IS the motor id
    hardware_interface::InterfaceInfo position_cmd;
    position_cmd.name = "position";
    joint.command_interfaces.push_back(position_cmd);
    hardware_interface::InterfaceInfo position_state;
    position_state.name = "position";
    joint.state_interfaces.push_back(position_state);
    hardware_interface::InterfaceInfo velocity_state;
    velocity_state.name = "velocity";
    joint.state_interfaces.push_back(velocity_state);
    info.joints.push_back(joint);
  }
  return info;
}

TEST(SO101HardwareAdapter, OnInitBuildsSdkConfigFromHardwareInfo)
{
  so101_hardware::SO101SystemHardware hw;
  const auto info = make_arm_info({
    {"port", "/dev/ttyTEST"},
    {"calib_file", "/tmp/test_calib.json"},
    {"reset_positions", R"({"1": 0.0, "2": -1.5})"},
  });
  EXPECT_EQ(
    hw.on_init(info), hardware_interface::CallbackReturn::SUCCESS);

  // The exported interfaces reflect the URDF joint structure.
  const auto states = hw.export_state_interfaces();
  ASSERT_EQ(states.size(), 12U);  // 6 joints × (position + velocity)
  EXPECT_EQ(states[0].get_prefix_name(), "1");
  EXPECT_EQ(states[0].get_interface_name(), "position");
  EXPECT_EQ(states[1].get_interface_name(), "velocity");
  const auto commands = hw.export_command_interfaces();
  ASSERT_EQ(commands.size(), 6U);
  EXPECT_EQ(commands[0].get_interface_name(), "position");
}

TEST(SO101HardwareAdapter, OnInitAppliesDefaultsWhenParamsAbsent)
{
  so101_hardware::SO101SystemHardware hw;
  EXPECT_EQ(
    hw.on_init(make_arm_info()), hardware_interface::CallbackReturn::SUCCESS);
}

TEST(SO101HardwareAdapter, RejectsInvalidUpdateRate)
{
  for (const auto * rate : {"0", "-1", "nan", "inf", "100garbage"}) {
    so101_hardware::SO101SystemHardware hw;
    EXPECT_EQ(hw.on_init(make_arm_info({{"update_rate", rate}})),
      hardware_interface::CallbackReturn::ERROR);
  }
}

TEST(SO101HardwareAdapter, MappedJointNamesRetainExplicitMotorIds)
{
  auto info = make_arm_info({}, {"shoulder", "gripper"});
  info.joints[0].parameters["id"] = "1";
  info.joints[1].parameters["id"] = "6";
  so101_hardware::SO101SystemHardware hw;
  EXPECT_EQ(hw.on_init(info), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.export_command_interfaces()[0].get_prefix_name(), "shoulder");
}

TEST(SO101HardwareAdapter, InvalidMappingInterfacesAndResetFailClosed)
{
  for (const auto & id : {"1garbage", "257", "-1"}) {
    auto info = make_arm_info();
    info.joints[0].parameters["id"] = id;
    so101_hardware::SO101SystemHardware hw;
    EXPECT_EQ(hw.on_init(info), hardware_interface::CallbackReturn::ERROR);
  }
  auto info = make_arm_info();
  info.joints[0].command_interfaces[0].name = "velocity";
  so101_hardware::SO101SystemHardware invalid_interface;
  EXPECT_EQ(invalid_interface.on_init(info), hardware_interface::CallbackReturn::ERROR);
  so101_hardware::SO101SystemHardware invalid_reset;
  EXPECT_EQ(invalid_reset.on_init(make_arm_info({{"reset_positions", "bad json"}})),
    hardware_interface::CallbackReturn::ERROR);
}

class TempCalibFile
{
public:
  TempCalibFile()
  {
    path_ = "/tmp/so101_hardware_test_" + std::to_string(getpid()) + ".json";
    std::ofstream file(path_);
    file << R"({
      "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "3": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "6": {"homing_offset": 0, "range_min": 0, "range_max": 4095}
    })";
  }
  ~TempCalibFile() {std::remove(path_.c_str());}
  const std::string & path() const {return path_;}

private:
  std::string path_;
};

class SO101HardwareSerial : public ::testing::Test
{
protected:
  void SetUp() override
  {
    master = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
    ASSERT_GE(master, 0);
    ASSERT_EQ(grantpt(master), 0);
    ASSERT_EQ(unlockpt(master), 0);
    const char * port = ptsname(master);
    ASSERT_NE(port, nullptr);
    ASSERT_EQ(hw.on_init(make_arm_info({
      {"calib_file", calib.path()}, {"update_rate", "100"}})),
      hardware_interface::CallbackReturn::SUCCESS);
    feetech::BusOptions options;
    options.port = port;
    for (std::uint8_t id = 1; id <= 6; ++id) {
      feetech::MotorConfig motor;
      motor.id = id;
      options.motors.push_back(motor);
    }
    auto bus = std::make_unique<feetech::Bus>(options);
    ASSERT_TRUE(bus->open());
    bus_attached = true;
    ASSERT_TRUE(so101_hardware::SO101HardwareTestAccess::attach_bus(hw, std::move(bus)));
  }

  void TearDown() override
  {
    if (bus_attached) {
      so101_hardware::SO101HardwareTestAccess::bus(hw).close();
    }
    if (master >= 0) {
      close(master);
    }
  }

  TempCalibFile calib;
  so101_hardware::SO101SystemHardware hw;
  int master = -1;
  bool bus_attached = false;
};

TEST_F(SO101HardwareSerial, DelayedFeedbackIsNotTruncatedByTheControlPeriod)
{
  std::thread responder([this]() {
      std::array<unsigned char, 14> request{};
      size_t received = 0;
      while (received < request.size()) {
        pollfd descriptor{master, POLLIN, 0};
        const int ready = poll(&descriptor, 1, 1000);
        if (ready < 0 && errno == EINTR) {
          continue;
        }
        ASSERT_GT(ready, 0);
        const auto count = ::read(master, request.data() + received, request.size() - received);
        ASSERT_GT(count, 0);
        received += static_cast<size_t>(count);
      }
      ASSERT_EQ(request[4], 0x82);  // SYNC_READ
      ASSERT_EQ(request[5], 56);
      ASSERT_EQ(request[6], 15);
      std::vector<unsigned char> replies;
      for (unsigned char id = 1; id <= 6; ++id) {
        std::vector<unsigned char> reply = {0xFF, 0xFF, id, 17, 0};
        reply.resize(20, 0);
        reply[6] = 9;  // Position = 2304 ticks; distinguish from held zero state.
        unsigned char checksum = 0;
        for (size_t i = 2; i < reply.size(); ++i) {
          checksum += reply[i];
        }
        reply.push_back(static_cast<unsigned char>(~checksum));
        replies.insert(replies.end(), reply.begin(), reply.end());
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      ASSERT_EQ(::write(master, replies.data(), replies.size()),
        static_cast<ssize_t>(replies.size()));
    });
  const auto result = hw.read(rclcpp::Time(0), rclcpp::Duration::from_seconds(0.01));
  responder.join();
  EXPECT_EQ(result, hardware_interface::return_type::OK);
  const auto states = hw.export_state_interfaces();
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_GT(states[i * 2].get_value(), 0.3);
  }
  EXPECT_EQ(so101_hardware::SO101HardwareTestAccess::arm(hw).health().fault,
    feetech::Fault::None);
}

class SO101HardwareFailures : public ::testing::TestWithParam<int>
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    ASSERT_EQ(hw.on_init(make_arm_info({
      {"simulated", "true"}, {"calib_file", calib.path()},
      {"update_rate", std::to_string(GetParam())}})),
      hardware_interface::CallbackReturn::SUCCESS);
    ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
    ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  }

  void TearDown() override
  {
    if (hw.on_shutdown(unused) != hardware_interface::CallbackReturn::SUCCESS) {
      ADD_FAILURE() << "hardware shutdown failed";
    }
    rclcpp::shutdown();
  }

  hardware_interface::return_type read()
  {
    // Frozen ROS time must not prevent steady-time failure escalation.
    return hw.read(rclcpp::Time(0), rclcpp::Duration::from_seconds(1.0 / GetParam()));
  }

  feetech::Bus::SimControl & sim()
  {return so101_hardware::SO101HardwareTestAccess::bus(hw).sim();}

  TempCalibFile calib;
  so101_hardware::SO101SystemHardware hw;
  rclcpp_lifecycle::State unused;
};

TEST_P(SO101HardwareFailures, InjectedTransientReadRecovers)
{
  sim().set_converge_step_ticks(0);
  sim().inject_sync_read_failure();
  EXPECT_EQ(read(), hardware_interface::return_type::OK);
  // Match LeRobot's default: one attempt, then recovery on the next cycle.
  EXPECT_EQ(so101_hardware::SO101HardwareTestAccess::bus(hw).health().consecutive_failures, 1U);
  sim().set_position_ticks(1, 2300);
  EXPECT_EQ(read(), hardware_interface::return_type::OK);
  const auto states = hw.export_state_interfaces();
  EXPECT_GT(states[0].get_value(), 0.1);

  // A deadline from a previous cycle must not poison a later activation.
  std::this_thread::sleep_for(std::chrono::milliseconds(45));
  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
}

TEST_P(SO101HardwareFailures, FailedReadHoldsAllValuesAndRecoveryResetsWindow)
{
  sim().set_converge_step_ticks(0);
  sim().set_position_ticks(1, 2300);
  ASSERT_EQ(read(), hardware_interface::return_type::OK);
  auto states = hw.export_state_interfaces();
  std::vector<double> values;
  for (const auto & state : states) {
    values.push_back(state.get_value());
  }
  const auto currents = so101_hardware::SO101HardwareTestAccess::currents(hw);
  sim().inject_sync_read_failure();
  sim().set_responsive(1, false);
  sim().set_position_ticks(2, 2600);
  ASSERT_EQ(read(), hardware_interface::return_type::OK);
  for (size_t i = 0; i < states.size(); ++i) {
    EXPECT_DOUBLE_EQ(states[i].get_value(), values[i]);
  }
  EXPECT_EQ(so101_hardware::SO101HardwareTestAccess::currents(hw), currents);

  std::this_thread::sleep_for(std::chrono::milliseconds(210));
  sim().set_responsive(1, true);
  ASSERT_EQ(read(), hardware_interface::return_type::OK);
  EXPECT_NE(states[2].get_value(), values[2]);
  sim().set_responsive(1, false);
  EXPECT_EQ(read(), hardware_interface::return_type::OK);
  sim().set_responsive(1, true);
}

TEST_P(SO101HardwareFailures, SustainedReadFailureErrorsAfter200Milliseconds)
{
  sim().inject_sync_read_failure();
  sim().set_responsive(1, false);
  const auto started = std::chrono::steady_clock::now();
  auto result = read();
  EXPECT_EQ(result, hardware_interface::return_type::OK);
  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(1.0 / GetParam()));
  auto next_cycle = started + period;
  while (result == hardware_interface::return_type::OK &&
      std::chrono::steady_clock::now() - started < std::chrono::seconds(1))
  {
    std::this_thread::sleep_until(next_cycle);
    result = read();
    next_cycle += period;
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_EQ(result, hardware_interface::return_type::ERROR);
  EXPECT_GE(elapsed, std::chrono::milliseconds(200));
  // One sample interval plus scheduling slack; 10 Hz must not wait 20 cycles.
  EXPECT_LT(elapsed, std::chrono::milliseconds(200) + period + std::chrono::milliseconds(100));
  sim().set_responsive(1, true);
  EXPECT_EQ(read(), hardware_interface::return_type::OK);
}

TEST_P(SO101HardwareFailures, MissingMotorAbortsActivationAndReleasesTorque)
{
  sim().set_responsive(3, false);
  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::ERROR);
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_FALSE(sim().torque_enabled(id));
  }
  sim().set_responsive(3, true);
  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
}

TEST_P(SO101HardwareFailures, WriteFailureUsesTimeWindowAndRecovers)
{
  auto & bus = so101_hardware::SO101HardwareTestAccess::bus(hw);
  bus.close();
  const auto write = [this]() {
      return hw.write(rclcpp::Time(0), rclcpp::Duration::from_seconds(1.0 / GetParam()));
    };
  EXPECT_EQ(write(), hardware_interface::return_type::OK);
  std::this_thread::sleep_for(std::chrono::milliseconds(210));
  EXPECT_EQ(write(), hardware_interface::return_type::ERROR);
  ASSERT_TRUE(bus.open());
  EXPECT_EQ(write(), hardware_interface::return_type::OK);
  bus.close();
  EXPECT_EQ(write(), hardware_interface::return_type::OK);
  ASSERT_TRUE(bus.open());
}

TEST_P(SO101HardwareFailures, InactiveWritesSkipRetriesAndLeaveEscalationToRead)
{
  auto & arm = so101_hardware::SO101HardwareTestAccess::arm(hw);
  auto commands = hw.export_command_interfaces();
  commands[0].set_value(1.0);
  const auto check_write = [this, &arm]() {
      const auto targets = arm.command_targets();
      const auto started = std::chrono::steady_clock::now();
      EXPECT_EQ(hw.write(rclcpp::Time(0), rclcpp::Duration::from_seconds(1.0)),
        hardware_interface::return_type::OK);
      EXPECT_LT(std::chrono::steady_clock::now() - started,
        so101_hardware::SO101HardwareTestAccess::write_budget(hw));
      EXPECT_EQ(arm.command_targets(), targets);
      EXPECT_EQ(sim().position_ticks(1), 2048);
    };

  // A failed release faults the arm while torque is still on: writes must not move it.
  sim().reject_register_write(1, 40, 0);
  ASSERT_FALSE(arm.stop(so101::StopPolicy::TorqueOff));
  ASSERT_EQ(arm.health().lifecycle, so101::Lifecycle::Faulted);
  check_write();
  EXPECT_EQ(read(), hardware_interface::return_type::OK);
  std::this_thread::sleep_for(std::chrono::milliseconds(210));
  check_write();
  EXPECT_EQ(read(), hardware_interface::return_type::ERROR);
  sim().clear_injections();

  ASSERT_TRUE(arm.deactivate());
  ASSERT_EQ(arm.health().lifecycle, so101::Lifecycle::Disconnected);
  check_write();
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(arm.health().lifecycle, so101::Lifecycle::Connected);
  check_write();
}

TEST_P(SO101HardwareFailures, ImmediateWriteFailuresBackOffWithinRetryBudget)
{
  auto & bus = so101_hardware::SO101HardwareTestAccess::bus(hw);
  bus.close();
  const auto budget = so101_hardware::SO101HardwareTestAccess::write_budget(hw);
  EXPECT_NEAR(std::chrono::duration<double>(budget).count(), 0.4 / GetParam(), 1e-8);
  struct timespec cpu_start{}, cpu_end{};
  ASSERT_EQ(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start), 0);
  const auto started = std::chrono::steady_clock::now();
  for (int attempt = 0; attempt < 3; ++attempt) {
    hw.write(rclcpp::Time(0), rclcpp::Duration::from_seconds(1.0 / GetParam()));
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_EQ(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end), 0);
  const auto cpu = std::chrono::seconds(cpu_end.tv_sec - cpu_start.tv_sec) +
    std::chrono::nanoseconds(cpu_end.tv_nsec - cpu_start.tv_nsec);
  EXPECT_LT(cpu, elapsed / 2);
  EXPECT_LT(elapsed, 3 * budget + std::chrono::milliseconds(30));
  ASSERT_TRUE(bus.open());
}

INSTANTIATE_TEST_SUITE_P(UpdateRates, SO101HardwareFailures, ::testing::Values(10, 100));

// The runtime TORQUE_OFF stop reaches the SDK through the hardware lifecycle:
// set_hardware_component_state(inactive) -> on_deactivate. Clearing the stop
// re-enters on_activate WITHOUT on_configure, so the bus must survive.
TEST(SO101HardwareAdapter, SimulatedLifecycleDeactivateReactivates)
{
  // on_activate creates an rclcpp node for the joint-current publisher.
  rclcpp::init(0, nullptr);
  TempCalibFile calib;
  so101_hardware::SO101SystemHardware hw;
  const auto info = make_arm_info({
    {"simulated", "true"},
    {"calib_file", calib.path()},
  });
  const rclcpp_lifecycle::State unused;
  ASSERT_EQ(hw.on_init(info), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_configure(unused), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);

  auto commands = hw.export_command_interfaces();
  auto states = hw.export_state_interfaces();
  commands[0].set_value(1.0);
  ASSERT_EQ(hw.write(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
  for (int i = 0; i < 20; ++i) {
    ASSERT_EQ(
      hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)),
      hardware_interface::return_type::OK);
  }
  EXPECT_NEAR(states[0].get_value(), 1.0, 0.01);

  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);
  // inactive -> active: no reconnect, torque re-enabled, reads fresh again
  EXPECT_EQ(hw.on_activate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_NEAR(commands[0].get_value(), 1.0, 0.01);
  EXPECT_EQ(hw.write(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
  EXPECT_EQ(hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);

  commands[0].set_value(-1.0);
  ASSERT_EQ(hw.write(rclcpp::Time(0), rclcpp::Duration(0, 0)), hardware_interface::return_type::OK);
    ASSERT_EQ(
      hw.perform_command_mode_switch({}, {"1/position"}),
      hardware_interface::return_type::OK);
  const double held = commands[0].get_value();
  EXPECT_GT(held, 0.0);  // stop during travel, not at the old -1.0 target
  for (int i = 0; i < 20; ++i) {
    ASSERT_EQ(
      hw.write(rclcpp::Time(0), rclcpp::Duration(0, 0)),
      hardware_interface::return_type::OK);
    ASSERT_EQ(
      hw.read(rclcpp::Time(0), rclcpp::Duration(0, 0)),
      hardware_interface::return_type::OK);
    EXPECT_NEAR(states[0].get_value(), held, 0.002);
  }

  EXPECT_EQ(hw.on_deactivate(unused), hardware_interface::CallbackReturn::SUCCESS);
  EXPECT_EQ(hw.on_cleanup(unused), hardware_interface::CallbackReturn::SUCCESS);
  rclcpp::shutdown();
}

}  // namespace
