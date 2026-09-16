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

#ifndef FEETECH__SIM_SMS_STS_HPP_
#define FEETECH__SIM_SMS_STS_HPP_

#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "SMS_STS.h"

namespace feetech
{

/// Simulated Feetech transport: an SMS_STS whose serial I/O is answered by an
/// in-memory motor simulation. All vendor protocol logic (frame encoding,
/// sync read/write, ACK parsing) runs for real; only the byte transport is
/// simulated. This lets the real Bus logic — protocol encoding, configuration
/// sequencing, rollback, failure classification — run headless in tests.
class SimSmsSts : public SMS_STS
{
public:
  struct Motor
  {
    bool present = true;
    bool responsive = true;   // answers reads
    bool acks_writes = true;  // answers ack'd writes
    std::uint8_t response_status = 0;
    int fail_next_writes = 0; // ack failures remaining (0 = always ack)
    bool torque_on = false;
    bool eprom_locked = true;
    bool wheel_mode = false;
    std::int32_t pos_ticks = 2048;      // present position (raw)
    std::int32_t goal_ticks = 2048;     // position command (raw)
    std::int32_t wheel_speed_steps = 0; // wheel velocity command (signed)
    std::int32_t last_move_ticks = 0;   // delta applied by the last read
    // Static register storage for write/readback of config registers.
    std::map<std::uint8_t, std::uint8_t> regs;
    // Register-precise write rejection: `rejected[addr]` is the set of first
    // data bytes to reject; an empty set rejects every write to the address.
    std::map<std::uint8_t, std::set<std::uint8_t>> rejected;
  };

  SimSmsSts();

  void add_motor(std::uint8_t id);
  Motor * motor(std::uint8_t id);
  std::map<std::uint8_t, Motor> & motors() { return motors_; }

  /// When true, the next synchronized group read queues no responses
  /// (transport-level failure).
  bool drop_next_sync_read = false;
  /// Position convergence applied per group read, in ticks.
  std::int32_t converge_step_ticks = 64;

protected:
  int writeSCS(unsigned char * nDat, int nLen) override;
  int writeSCS(unsigned char bDat) override;
  int readSCS(unsigned char * nDat, int nLen) override;
  int readSCS(unsigned char * nDat, int nLen, unsigned long TimeOut) override;
  void rFlushSCS() override;
  void wFlushSCS() override;

private:
  std::map<std::uint8_t, Motor> motors_;
  std::vector<std::uint8_t> rx_;     // pending response bytes
  std::vector<std::uint8_t> frame_;  // frame accumulator

  void on_frame(const std::vector<std::uint8_t> & frame);
  void queue_bytes(const std::vector<std::uint8_t> & bytes);
  void queue_ack(std::uint8_t id);
  void advance_time();
  bool write_register(std::uint8_t id, std::uint8_t addr, const std::uint8_t * data, int len);
  std::vector<std::uint8_t> register_block(
    std::uint8_t id, std::uint8_t addr, int len) const;
};

}  // namespace feetech

#endif  // FEETECH__SIM_SMS_STS_HPP_
