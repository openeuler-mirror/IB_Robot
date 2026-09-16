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

#ifndef FEETECH__BUS_HPP_
#define FEETECH__BUS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "feetech/types.hpp"

namespace feetech
{

/// Bus construction options.
struct BusOptions
{
  /// Serial device (ignored when `simulated` is true).
  std::string port;
  /// Bus baudrate (proven Feetech default: 1 Mbps).
  std::uint32_t baudrate = 1'000'000;
  /// Motor registry. Position and wheel groups are derived from each
  /// MotorConfig::mode.
  std::vector<MotorConfig> motors;
  /// Construct against a simulated transport that behaves like the real bus
  /// (see Bus::sim for failure injection and state control).
  bool simulated = false;
};

/// A Feetech STS/SMS motor bus.
///
/// Owns the serial transport (or its simulation), the motor registry, and bus
/// health. All state and command values are SI; raw servo units never escape
/// this class. All I/O reports failures explicitly — a failed read never
/// presents cached values as fresh.
class Bus
{
public:
  /// Test/inspection handle for the simulated transport. Only valid when the
  /// bus was constructed with BusOptions::simulated.
  class SimControl
  {
  public:
    /// Make the next synchronized group read fail at the transport level
    /// (no motor responses are queued for the round trip).
    void inject_sync_read_failure();
    void clear_sync_read_failure();
    /// Make a motor stop answering synchronized group reads.
    void set_responsive(std::uint8_t id, bool responsive);
    /// Make a motor stop acknowledging ack'd writes (config, torque).
    void set_write_ack(std::uint8_t id, bool acks);
    /// Status flags carried by every response from this motor (0 = healthy).
    void set_response_status(std::uint8_t id, std::uint8_t status);
    /// Fail the next `count` ack'd writes to a motor (0 = always ack).
    void fail_next_writes(std::uint8_t id, int count);
    /// Reject every write to a register address (motor NACKs; used to fail a
    /// specific configuration step).
    void reject_register_write(std::uint8_t id, std::uint8_t addr);
    /// Reject writes to a register address whose first data byte equals
    /// `value` (e.g. reject torque-enable but allow torque-disable).
    void reject_register_write(std::uint8_t id, std::uint8_t addr, std::uint8_t value);
    void clear_injections();
    /// Seed a motor's present position in raw ticks.
    void set_position_ticks(std::uint8_t id, std::int32_t ticks);
    /// Read a motor's present position in raw ticks (0 when unknown).
    std::int32_t position_ticks(std::uint8_t id) const;
    /// Position convergence applied per group read, in ticks.
    void set_converge_step_ticks(std::int32_t step);
    /// Motor state inspection (torque enabled / EPROM locked).
    bool torque_enabled(std::uint8_t id) const;
    bool eprom_locked(std::uint8_t id) const;

  private:
    friend class Bus;
    SimControl();
    class Impl;
    std::shared_ptr<Impl> impl_;
  };

  explicit Bus(BusOptions options);
  ~Bus();

  Bus(const Bus &) = delete;
  Bus & operator=(const Bus &) = delete;

  /// --- lifecycle -----------------------------------------------------------

  /// Open the bus. Real mode opens the serial port; simulated mode always
  /// succeeds. Returns false (and records Fault::PortOpenFailed) when the
  /// port cannot be opened.
  bool open();

  /// Close the bus. Idempotent.
  void close();

  bool is_open() const;

  /// Probe all registered motors with bounded retries, without register writes.
  MotorOpResult ping_all(int retries = 3, std::uint32_t retry_gap_us = 10000);

  /// Bound real transport reads and writes across a control cycle. Clear before
  /// lifecycle operations; the bus must be used by only one caller at a time.
  void set_io_deadline(std::chrono::steady_clock::time_point deadline);
  void clear_io_deadline();

  /// --- configuration -------------------------------------------------------

  /// Apply one motor's configuration atomically: torque off, EPROM unlock,
  /// calibration + gains, EPROM lock, torque on. On failure the motor is left
  /// torque-disabled with EPROM locked when rollback succeeds. Incomplete
  /// torque release reports EmergencyPartiallyFailed; relock failures are detailed.
  MotorOpResult apply_motor_config(const MotorConfig & config);

  /// Apply every registered motor's configuration in registry order,
  /// aborting and rolling back all previously configured motors on the first
  /// failure (rollback failures are reported as for apply_motor_config).
  MotorOpResult apply_all_configs();

  /// Apply configurations for the given motors only (registry order among
  /// the given ids). Shared-bus composition: a subsystem configures its own
  /// motors without touching others.
  MotorOpResult apply_configs(const std::vector<std::uint8_t> & ids);

  /// Read-only check that position-mode firmware calibration matches the
  /// registered configuration. Never writes registers or changes torque.
  MotorOpResult verify_calibration();

  /// --- realtime I/O --------------------------------------------------------

  /// LeRobot Feetech receive budget: response wire time + 3 byte times + 50 ms,
  /// rounded up to the vendor's millisecond resolution for the requested group.
  std::chrono::milliseconds sync_read_timeout(std::size_t motor_count) const;

  /// Synchronized group read of every registered motor. On success `out`
  /// holds one valid sample per motor in registry order. On failure `out` is
  /// cleared and the result identifies the transport failure or the first
  /// missing motor.
  MotorOpResult sync_read(std::vector<MotorSample> & out);

  /// Synchronized group read of the given motors only (samples in the given
  /// order). Shared-bus composition: a subsystem reads its own motors;
  /// unresponsive motors outside the subset do not fail the read.
  MotorOpResult sync_read(std::vector<MotorSample> & out, const std::vector<std::uint8_t> & ids);

  /// Synchronized position command for position-mode motors (radians).
  /// Targets for unknown ids or wheel-mode motors are rejected. Targets
  /// outside the tick range are clamped and reported in `result.clamped`.
  MotorOpResult sync_write_positions(const std::vector<MotorTarget> & targets);

  /// Synchronized velocity command for wheel-mode motors (rad/s). Targets
  /// for unknown ids or position-mode motors are rejected.
  MotorOpResult sync_write_velocities(const std::vector<MotorTarget> & targets);

  /// --- emergency -----------------------------------------------------------

  /// Command torque-off for every registered motor with retries. Reports
  /// EmergencyPartiallyFailed listing any motor that could not be released.
  MotorOpResult emergency_release_all();

  /// Command torque-off for the given motors only. Shared-bus composition:
  /// a subsystem releases its own motors without affecting others.
  MotorOpResult emergency_release(const std::vector<std::uint8_t> & ids);

  /// --- introspection -------------------------------------------------------

  const BusHealth & health() const;
  const std::vector<MotorConfig> & motors() const;

  /// Simulation control handle. Aborts when the bus is not simulated.
  SimControl & sim();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace feetech

#endif  // FEETECH__BUS_HPP_
