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

// Minimal pybind11 bindings for so101_sdk: exposes the Arm (follower) and
// LeaderArm for Python consumers (so101_backend, teleop, calibration tools).
// The feetech bus protocol stays entirely in C++; Python sees only the
// SDK's SI-unit, named-joint API surface.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "so101/arm.hpp"
#include "so101/leader.hpp"

namespace py = pybind11;
using namespace so101;

namespace
{

// ArmState as a plain Python dict: {joint_name: {"position": ..., "velocity":
// ..., "effort": ...}} — no C++ types leak into Python.
py::dict arm_state_to_dict(const ArmState & state)
{
  py::dict result;
  for (const auto & [name, reading] : state.joints) {
    py::dict entry;
    entry["position"] = reading.position;
    entry["velocity"] = reading.velocity;
    entry["effort"] = reading.effort;
    result[py::str(name)] = entry;
  }
  return result;
}

}  // namespace

PYBIND11_MODULE(so101_sdk_py, m)
{
  m.doc() = "Minimal Python bindings for so101_sdk (Arm, LeaderArm)";

  // --- Enums must be registered before classes that use them as defaults --
  py::enum_<so101::Lifecycle>(m, "Lifecycle")
    .value("Disconnected", so101::Lifecycle::Disconnected)
    .value("Connected", so101::Lifecycle::Connected)
    .value("Activated", so101::Lifecycle::Activated)
    .value("Faulted", so101::Lifecycle::Faulted);

  py::enum_<StopPolicy>(m, "StopPolicy")
    .value("HoldLast", StopPolicy::HoldLast)
    .value("TorqueOff", StopPolicy::TorqueOff);

  // --- ArmConfig ------------------------------------------------------------
  py::class_<ArmConfig>(m, "ArmConfig")
    .def(py::init<>())
    .def_readwrite("port", &ArmConfig::port)
    .def_readwrite("baudrate", &ArmConfig::baudrate)
    .def_readwrite("calibration_file", &ArmConfig::calibration_file)
    .def_readwrite("joint_order", &ArmConfig::joint_order)
    .def_readwrite("motor_ids", &ArmConfig::motor_ids)
    .def_readwrite("reset_positions", &ArmConfig::reset_positions)
    .def_readwrite("simulated", &ArmConfig::simulated);

  // --- ArmHealth (simplified as a dict) -----------------------------------------
  py::class_<ArmHealth>(m, "ArmHealth")
    .def(py::init<>())
    .def_readonly("lifecycle", &ArmHealth::lifecycle)
    .def_property_readonly("fault", [](const ArmHealth & health) { return static_cast<int>(health.fault); })
    .def_readonly("detail", &ArmHealth::detail);

  // --- Arm (follower) -------------------------------------------------------------
  py::class_<Arm>(m, "Arm")
    .def(py::init<ArmConfig>(), py::arg("config"))
    .def("connect", &Arm::connect)
    .def("activate", &Arm::activate)
    .def("read", [](Arm & self) -> py::object {
        ArmState state;
        if (!self.read(state)) {
            return py::none();
        }
        return arm_state_to_dict(state);
    })
    .def("write_targets", [](Arm & self, const std::map<std::string, double> & targets) {
        return self.write_targets(targets);
    })
    .def("hold", &Arm::hold)
    .def("stop", &Arm::stop, py::arg("policy") = StopPolicy::HoldLast)
    .def("deactivate", &Arm::deactivate)
    .def("calibrated_ranges", &Arm::calibrated_ranges)
    .def("health", &Arm::health)
    .def("joint_names", &Arm::joint_names)
    .def("sim", [](Arm & self) -> feetech::Bus::SimControl {
        return self.sim();
    });

  // --- LeaderConfig / LeaderArm ------------------------------------------------------
  py::class_<LeaderConfig>(m, "LeaderConfig")
    .def(py::init<>())
    .def_readwrite("port", &LeaderConfig::port)
    .def_readwrite("baudrate", &LeaderConfig::baudrate)
    .def_readwrite("calibration_file", &LeaderConfig::calibration_file)
    .def_readwrite("calibration_version", &LeaderConfig::calibration_version)
    .def_readwrite("joint_order", &LeaderConfig::joint_order)
    .def_readwrite("motor_ids", &LeaderConfig::motor_ids)
    .def_readwrite("gripper_joint", &LeaderConfig::gripper_joint)
    .def_readwrite("simulated", &LeaderConfig::simulated);

  py::class_<LeaderArm>(m, "LeaderArm")
    .def(py::init<LeaderConfig>(), py::arg("config"))
    .def("connect", &LeaderArm::connect)
    .def("disconnect", &LeaderArm::disconnect)
    .def("read", [](LeaderArm & self) -> py::object {
        ArmState state;
        if (!self.read(state)) {
            return py::none();
        }
        return arm_state_to_dict(state);
    })
    .def("health", &LeaderArm::health)
    .def("sim", [](LeaderArm & self) -> feetech::Bus::SimControl { return self.sim(); });

  // --- SimControl (test/inspection handle) ----------------------------------------------
  py::class_<feetech::Bus::SimControl>(m, "SimControl")
    .def("inject_sync_read_failure", &feetech::Bus::SimControl::inject_sync_read_failure)
    .def("clear_sync_read_failure", &feetech::Bus::SimControl::clear_sync_read_failure)
    .def("set_responsive", &feetech::Bus::SimControl::set_responsive)
    .def("set_write_ack", &feetech::Bus::SimControl::set_write_ack)
    .def("fail_next_writes", &feetech::Bus::SimControl::fail_next_writes)
    .def("clear_injections", &feetech::Bus::SimControl::clear_injections)
    .def("set_position_ticks", &feetech::Bus::SimControl::set_position_ticks)
    .def("position_ticks", &feetech::Bus::SimControl::position_ticks)
    .def("set_converge_step_ticks", &feetech::Bus::SimControl::set_converge_step_ticks)
    .def("torque_enabled", &feetech::Bus::SimControl::torque_enabled)
    .def("eprom_locked", &feetech::Bus::SimControl::eprom_locked);
}
