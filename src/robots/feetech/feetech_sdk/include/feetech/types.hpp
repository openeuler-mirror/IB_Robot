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

#ifndef FEETECH__TYPES_HPP_
#define FEETECH__TYPES_HPP_

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace feetech
{

/// Operating mode of a motor on the bus.
enum class Mode : std::uint8_t
{
  /// Position servo mode (STS3215 default).
  Position,
  /// Continuous rotation / wheel mode.
  Wheel,
};

/// Calibration and configuration for one motor.
///
/// Tick-valued fields are in raw servo units and are written to the motor
/// during configuration; they never appear in SDK state or command values,
/// which are SI (see conversion.hpp).
struct MotorConfig
{
  /// Bus ID of the motor.
  std::uint8_t id = 0;
  /// Canonical joint name (may be empty for anonymous groups).
  std::string name;
  /// Homing offset in ticks (sign-magnitude encoded on write).
  int homing_offset = 0;
  /// Calibrated minimum position in ticks.
  int range_min = 0;
  /// Calibrated maximum position in ticks.
  int range_max = 4095;
  /// Operating mode.
  Mode mode = Mode::Position;
  /// Position loop gains (register defaults match the proven SO-101 values).
  std::uint8_t kp = 16;
  std::uint8_t kd = 32;
  std::uint8_t ki = 0;
  /// Motion profile written with every position command.
  std::uint16_t profile_speed = 2400;
  std::uint8_t profile_acc = 50;
};

/// One motor's feedback converted to SI units.
struct MotorSample
{
  std::uint8_t id = 0;
  /// Radians. Position mode: centered on the calibration zero.
  /// Wheel mode: accumulated rotation.
  double position = 0.0;
  /// Radians per second.
  double velocity = 0.0;
  /// Amperes (motor current estimate).
  double effort = 0.0;
  /// True only when this sample came from a successful group read.
  bool valid = false;
};

/// Command for one motor.
///
/// Position mode uses `position` (radians); wheel mode uses `velocity`
/// (radians per second).
struct MotorTarget
{
  std::uint8_t id = 0;
  double position = 0.0;
  double velocity = 0.0;
};

/// Fault classification for health reporting.
enum class Fault : std::uint8_t
{
  None = 0,
  /// open() failed on the serial port.
  PortOpenFailed,
  /// I/O attempted before open().
  NotOpen,
  /// I/O attempted after close().
  Closed,
  /// Synchronized group read transmit/reply failed at the transport level.
  SyncReadFailed,
  /// A motor did not answer a synchronized group read.
  MotorMissing,
  /// A write was rejected before transmission (unknown id or wrong mode).
  WriteRejected,
  /// Motor configuration sequence failed (rolled back).
  ConfigFailed,
  /// Emergency torque release could not reach every motor.
  EmergencyPartiallyFailed,
};

/// Bus health information.
struct BusHealth
{
  Fault fault = Fault::None;
  /// Consecutive failed I/O operations (reset on the next success).
  std::uint32_t consecutive_failures = 0;
  /// Time of the last successful I/O operation.
  std::chrono::steady_clock::time_point last_ok{};
  /// False until the first successful I/O operation.
  bool ever_ok = false;
};

/// Outcome of an operation that can fail.
struct MotorOpResult
{
  bool ok = false;
  /// First offending motor id (0 when not applicable).
  std::uint8_t failed_id = 0;
  Fault fault = Fault::None;
  /// Human-readable context for logging.
  std::string detail;
  /// Position targets that were clamped to the tick range (id list).
  std::vector<std::uint8_t> clamped;
};

}  // namespace feetech

#endif  // FEETECH__TYPES_HPP_
