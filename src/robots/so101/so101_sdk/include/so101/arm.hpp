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

#ifndef SO101__ARM_HPP_
#define SO101__ARM_HPP_

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "feetech/bus.hpp"
#include "feetech/types.hpp"
#include "so101/calibration.hpp"

namespace so101
{

/// Follower arm construction options.
struct ArmConfig
{
  /// Serial device (ignored when `simulated` is true).
  std::string port;
  std::uint32_t baudrate = 1'000'000;
  /// Calibration file path (SO-101 JSON format, owned by this SDK).
  std::string calibration_file;
  /// Legacy explicit startup motion override, NOT semantic HOME. Omit to
  /// hold the measured pose. New runtimes keep HOME in public model metadata.
  std::map<std::string, double> reset_positions;
  /// Canonical joint order. SO-101 joint names are numeric motor ids.
  std::vector<std::string> joint_order = {"1", "2", "3", "4", "5", "6"};
  std::map<std::string, int> motor_ids;
  /// Construct against the simulated transport (for tests).
  bool simulated = false;
};

/// One joint's SI reading.
struct JointReading
{
  double position = 0.0;  // rad
  double velocity = 0.0;  // rad/s
  double effort = 0.0;    // A
};

/// Named-joint arm state with a single read timestamp. Callers never depend
/// on array ordering.
struct ArmState
{
  std::map<std::string, JointReading> joints;
  std::chrono::steady_clock::time_point stamp{};
};

enum class Lifecycle : std::uint8_t
{
  Disconnected,
  Connected,
  Activated,
  Faulted,
};

struct ArmHealth
{
  Lifecycle lifecycle = Lifecycle::Disconnected;
  feetech::Fault fault = feetech::Fault::None;
  std::string detail;
};

/// Safe-stop policy for Arm::stop().
enum class StopPolicy : std::uint8_t
{
  /// Re-apply the last commanded positions (position-servo default).
  HoldLast,
  /// Disable torque on every motor.
  TorqueOff,
};

/// SO-101 follower arm: owns its Feetech bus and calibration, exposes a
/// lifecycle (connect → activate → read/write/hold/stop → deactivate) and
/// SI-unit named-joint state. Failed reads never present stale values as
/// fresh; activation either completes or aborts into a safe state.
class Arm
{
public:
  explicit Arm(ArmConfig config);
  ~Arm();

  Arm(const Arm &) = delete;
  Arm & operator=(const Arm &) = delete;

  /// Open the bus and load/validate the calibration. Fails (with the file
  /// path and joint in `health().detail`) on any calibration problem.
  bool connect();

  /// Attach to a caller-owned shared bus (composition, e.g. LeKiwi mobile
  /// manipulator: one bus carries arm + wheel motors). The calibration must
  /// already be loaded/validated by the caller for the shared bus's motor
  /// registry. The arm does NOT own or close the bus in this mode.
  bool attach_shared_bus(feetech::Bus & bus, const Calibration & calibration);

  /// Apply motor configurations and seed initial commands from measured
  /// positions (or configured reset positions). On any failure the arm is
  /// left in a safe state (torque disabled) and activation reports failure.
  bool activate();

  /// Synchronized read of every joint while connected or activated.
  /// On failure `out` is cleared and the
  /// return value is false; stale values are never returned as fresh.
  bool read(ArmState & out);

  /// Command positions in radians by joint name. Unknown joint names fail.
  bool write_targets(const std::map<std::string, double> & targets);

  /// Re-apply the last commanded positions.
  bool hold();

  /// Safe stop per the given policy (default: hold last commanded state).
  /// TorqueOff releases torque and returns the lifecycle to Connected, so a
  /// subsequent activate() re-enables the arm without reconnecting.
  bool stop(StopPolicy policy = StopPolicy::HoldLast);

  /// Torque off and close the bus. Idempotent.
  bool deactivate();

  /// Calibrated position ranges in radians, by joint name (for LeRobot-style
  /// normalization tables).
  std::map<std::string, std::pair<double, double>> calibrated_ranges() const;

  const ArmHealth & health() const;
  const std::vector<std::string> & joint_names() const;
  const std::map<std::string, double> & command_targets() const { return last_targets_; }

  /// Simulated-transport control handle (aborts when not simulated).
  feetech::Bus::SimControl & sim();

private:
  bool write_targets_unlocked(const std::map<std::string, double> & targets);

  ArmConfig config_;
  /// Owns the bus after connect(), or points at a caller-owned shared bus
  /// after attach_shared_bus() (owns_bus_ = false).
  std::unique_ptr<feetech::Bus> owned_bus_;
  feetech::Bus * bus_ = nullptr;
  bool owns_bus_ = false;
  Calibration calibration_;
  ArmHealth health_;
  std::map<std::string, std::uint8_t> id_of_;
  /// Motor ids in joint_order order; scopes shared-bus operations.
  std::vector<std::uint8_t> my_ids_;
  std::map<std::string, double> last_targets_;
};

}  // namespace so101

#endif  // SO101__ARM_HPP_
