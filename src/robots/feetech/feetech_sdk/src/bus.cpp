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

#include "feetech/bus.hpp"

#include <SMS_STS.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "feetech/conversion.hpp"
#include "feetech/detail/rollback.hpp"
#include "safe_sms_sts.hpp"
#include "sim_sms_sts.hpp"

namespace feetech
{

namespace
{

// Extended feedback frame: present position, speed, current (SO-101 proven).
constexpr std::uint8_t kFeedbackStartAddr = SMS_STS_PRESENT_POSITION_L;
constexpr int kFeedbackLen = SMS_STS_PRESENT_CURRENT_H - SMS_STS_PRESENT_POSITION_L + 1;
constexpr double kSyncReadLatencyMs = 50.0;
constexpr int kTorqueRetryCount = 3;

// Register addresses used by the configuration sequence.
constexpr std::uint8_t kRegResponseDelay = 7;
constexpr std::uint8_t kRegP = 21;
constexpr std::uint8_t kRegD = 22;
constexpr std::uint8_t kRegI = 23;
constexpr std::uint8_t kRegHomingOfs = 31;

MotorOpResult make_failure(Fault fault, std::uint8_t id, std::string detail)
{
  MotorOpResult result;
  result.ok = false;
  result.fault = fault;
  result.failed_id = id;
  result.detail = std::move(detail);
  return result;
}

const char * validate_motor_config(const MotorConfig & config)
{
  if (config.id == 0 || config.id >= 254) {
    return "motor id must be within [1, 253]";
  }
  if (config.homing_offset < -2047 || config.homing_offset > 2047) {
    return "homing_offset must be within [-2047, 2047]";
  }
  if (config.range_min < 0 || config.range_max > 4095 || config.range_min > config.range_max) {
    return "range must satisfy 0 <= range_min <= range_max <= 4095";
  }
  return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// SimControl
// ---------------------------------------------------------------------------

class Bus::SimControl::Impl
{
public:
  explicit Impl(SimSmsSts * sim) : sim(sim) {}
  SimSmsSts * sim;
};

Bus::SimControl::SimControl() = default;

void Bus::SimControl::inject_sync_read_failure()
{
  if (impl_ && impl_->sim) {
    impl_->sim->drop_next_sync_read = true;
  }
}

void Bus::SimControl::clear_sync_read_failure()
{
  if (impl_ && impl_->sim) {
    impl_->sim->drop_next_sync_read = false;
  }
}

void Bus::SimControl::set_responsive(std::uint8_t id, bool responsive)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->responsive = responsive;
    }
  }
}

void Bus::SimControl::set_write_ack(std::uint8_t id, bool acks)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->acks_writes = acks;
    }
  }
}

void Bus::SimControl::set_response_status(std::uint8_t id, std::uint8_t status)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->response_status = status;
    }
  }
}

void Bus::SimControl::fail_next_writes(std::uint8_t id, int count)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->fail_next_writes = count;
    }
  }
}

void Bus::SimControl::reject_register_write(std::uint8_t id, std::uint8_t addr)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->rejected[addr] = {};  // empty set: reject every write to the address
    }
  }
}

void Bus::SimControl::reject_register_write(std::uint8_t id, std::uint8_t addr, std::uint8_t value)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->rejected[addr].insert(value);
    }
  }
}

void Bus::SimControl::clear_injections()
{
  if (!impl_ || !impl_->sim) {
    return;
  }
  impl_->sim->drop_next_sync_read = false;
  for (auto & [id, m] : impl_->sim->motors()) {
    (void)id;
    m.responsive = true;
    m.acks_writes = true;
    m.response_status = 0;
    m.fail_next_writes = 0;
    m.rejected.clear();
  }
}

void Bus::SimControl::set_position_ticks(std::uint8_t id, std::int32_t ticks)
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      m->pos_ticks = ticks;
      m->goal_ticks = ticks;
    }
  }
}

std::int32_t Bus::SimControl::position_ticks(std::uint8_t id) const
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      return m->pos_ticks;
    }
  }
  return 0;
}

void Bus::SimControl::set_converge_step_ticks(std::int32_t step)
{
  if (impl_ && impl_->sim) {
    impl_->sim->converge_step_ticks = step;
  }
}

bool Bus::SimControl::torque_enabled(std::uint8_t id) const
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      return m->torque_on;
    }
  }
  return false;
}

bool Bus::SimControl::eprom_locked(std::uint8_t id) const
{
  if (impl_ && impl_->sim) {
    if (auto * m = impl_->sim->motor(id)) {
      return m->eprom_locked;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Bus
// ---------------------------------------------------------------------------

class Bus::Impl
{
public:
  explicit Impl(BusOptions opts) : options(std::move(opts))
  {
    if (options.baudrate == 0) {
      throw std::invalid_argument("BusOptions.baudrate must be positive");
    }
    if (options.motors.empty() || options.motors.size() > 30) {
      throw std::invalid_argument("BusOptions.motors must contain 1..30 motors (serial frame capacity)");
    }
    std::unordered_map<std::uint8_t, bool> seen;
    for (const auto & m : options.motors) {
      if (const char * why = validate_motor_config(m)) {
        throw std::invalid_argument("invalid motor configuration: " + std::to_string(m.id) + ": " + why);
      }
      if (seen[m.id]) {
        throw std::invalid_argument("duplicate motor id in BusOptions: " + std::to_string(m.id));
      }
      seen[m.id] = true;
      (m.mode == Mode::Wheel ? wheel_ids : position_ids).push_back(m.id);
      config_by_id[m.id] = m;
    }
  }

  BusOptions options;
  BusHealth health;
  bool opened = false;
  std::size_t sync_read_sized_for = 0;

  std::unique_ptr<SafeSmsSts> real;
  std::unique_ptr<SimSmsSts> sim;
  std::unique_ptr<SimControl> sim_control;
  SMS_STS * proto = nullptr;

  std::vector<std::uint8_t> position_ids;
  std::vector<std::uint8_t> wheel_ids;
  std::unordered_map<std::uint8_t, MotorConfig> config_by_id;

  std::vector<std::uint8_t> all_ids() const
  {
    std::vector<std::uint8_t> ids = position_ids;
    ids.insert(ids.end(), wheel_ids.begin(), wheel_ids.end());
    return ids;
  }

  const MotorConfig * config_of(std::uint8_t id) const
  {
    auto it = config_by_id.find(id);
    return it == config_by_id.end() ? nullptr : &it->second;
  }

  void note_failure(Fault fault)
  {
    health.fault = fault;
    ++health.consecutive_failures;
  }

  void note_success()
  {
    health.fault = Fault::None;
    health.consecutive_failures = 0;
    health.last_ok = std::chrono::steady_clock::now();
    health.ever_ok = true;
  }

  // One motor's configuration sequence. `unlocked` and `configured` are the
  // rollback bookkeeping shared across apply_all_configs. On failure every
  // touched motor (the fully configured ones plus the failing one) is rolled
  // back: fail-closed torque disable, then best-effort EPROM relock.
  MotorOpResult apply_one(
    const MotorConfig & cfg, std::set<std::uint8_t> & unlocked,
    std::vector<std::uint8_t> & configured)
  {
    const std::uint8_t id = cfg.id;
    const auto require = [this](bool ok, const char * step) {
        if (!ok || proto->getState() != 0) {
          throw std::runtime_error(std::string(step));
        }
      };
    const auto step_gap = [] { std::this_thread::sleep_for(std::chrono::milliseconds(2)); };

    try {
      require(proto->EnableTorque(id, 0) != 0, "disable torque");
      step_gap();
      require(proto->unLockEprom(id) != 0, "unlock EPROM");
      unlocked.insert(id);
      step_gap();
      require(
        proto->writeWord(id, kRegHomingOfs, encode_homing_offset(cfg.homing_offset)) != 0,
        "write homing offset");
      require(proto->writeWord(id, 9, cfg.range_min) != 0, "write minimum range");
      require(proto->writeWord(id, 11, cfg.range_max) != 0, "write maximum range");
      require(proto->writeByte(id, kRegResponseDelay, 0) != 0, "write response delay");
      require(proto->writeByte(id, kRegP, cfg.kp) != 0, "write position P gain");
      require(proto->writeByte(id, kRegD, cfg.kd) != 0, "write position D gain");
      require(proto->writeByte(id, kRegI, cfg.ki) != 0, "write position I gain");
      if (cfg.mode == Mode::Wheel) {
        require(proto->WheelMode(id) != 0, "set wheel mode");
      } else {
        require(proto->writeByte(id, SMS_STS_MODE, 0) != 0, "set position mode");
      }
      step_gap();
      require(proto->LockEprom(id) != 0, "lock EPROM");
      unlocked.erase(id);
      step_gap();
      // Never energize a motor against a retained goal from an earlier
      // session. Seed an acknowledged hold command while torque is off.
      if (cfg.mode == Mode::Position) {
        const int ticks = proto->readWord(id, SMS_STS_PRESENT_POSITION_L);
        require(ticks >= 0 && ticks <= 4095 && proto->getState() == 0, "read activation position");
        require(proto->WritePosEx(id, ticks, cfg.profile_speed, cfg.profile_acc) != 0 &&
          proto->getState() == 0, "seed activation hold");
      } else {
        require(proto->WriteSpe(id, 0, cfg.profile_acc) != 0, "seed zero velocity");
      }
      require(proto->EnableTorque(id, 1) != 0, "enable torque");
      configured.push_back(id);
      MotorOpResult ok;
      ok.ok = true;
      return ok;
    } catch (const std::runtime_error & e) {
      std::vector<std::uint8_t> touched = configured;
      touched.push_back(id);
      const auto rollback = detail::rollback_partial_config(
        touched, unlocked,
        [this](std::uint8_t mid, std::uint8_t enable) {
          return proto->EnableTorque(mid, enable) != 0 && proto->getState() == 0;
        },
        [this](std::uint8_t mid) {
          return proto->LockEprom(mid) != 0 && proto->getState() == 0;
        }, kTorqueRetryCount);
      std::string message = std::string("configuration step failed: ") + e.what();
      if (!rollback.eprom_relocked_all) {
        message += "; EPROM left unlocked on motors:";
        for (const auto mid : rollback.relock_failures) {
          message += " " + std::to_string(mid);
        }
      }
      if (!rollback.torque_disabled_all) {
        message += "; ROLLBACK INCOMPLETE: torque could not be disabled on every touched motor";
        return make_failure(Fault::EmergencyPartiallyFailed, id, message);
      }
      return make_failure(Fault::ConfigFailed, id, message);
    }
  }
};

Bus::Bus(BusOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

Bus::~Bus() { close(); }

bool Bus::open()
{
  if (impl_->opened) {
    return true;
  }
  if (impl_->options.simulated) {
    impl_->sim = std::make_unique<SimSmsSts>();
    for (const auto & m : impl_->options.motors) {
      impl_->sim->add_motor(m.id);
    }
    impl_->proto = impl_->sim.get();
    auto sim_impl = std::make_shared<SimControl::Impl>(impl_->sim.get());
    impl_->sim_control.reset(new SimControl());
    impl_->sim_control->impl_ = sim_impl;
  } else {
    impl_->real = std::make_unique<SafeSmsSts>();
    if (!impl_->real->begin(
        static_cast<int>(impl_->options.baudrate), impl_->options.port.c_str()))
    {
      impl_->note_failure(Fault::PortOpenFailed);
      impl_->real.reset();
      return false;
    }
    impl_->proto = impl_->real.get();
  }
  const auto ids = impl_->all_ids();
  impl_->proto->syncReadBegin(
    static_cast<std::uint8_t>(ids.size()), kFeedbackLen,
    static_cast<std::uint32_t>(sync_read_timeout(ids.size()).count()));
  impl_->sync_read_sized_for = ids.size();
  impl_->opened = true;
  return true;
}

void Bus::close()
{
  if (!impl_->opened) {
    return;
  }
  // Vendor syncReadBegin allocates an array but syncReadEnd uses scalar
  // delete. Keep the pinned source intact and release its public buffer here.
  delete[] impl_->proto->syncReadRxBuff;
  impl_->proto->syncReadRxBuff = nullptr;
  if (impl_->sim_control) {
    impl_->sim_control->impl_->sim = nullptr;
  }
  if (impl_->real) {
    impl_->real->end();
    impl_->real.reset();
  }
  impl_->sim.reset();
  impl_->sim_control.reset();
  impl_->proto = nullptr;
  impl_->opened = false;
}

bool Bus::is_open() const { return impl_->opened; }

void Bus::set_io_deadline(std::chrono::steady_clock::time_point deadline)
{
  if (impl_->real) {
    impl_->real->set_io_deadline(deadline);
  }
}

void Bus::clear_io_deadline()
{
  if (impl_->real) {
    impl_->real->clear_io_deadline();
  }
}

MotorOpResult Bus::ping_all(int retries, std::uint32_t retry_gap_us)
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  if (retries <= 0) {
    return make_failure(Fault::WriteRejected, 0, "ping retries must be positive");
  }
  for (const auto & cfg : impl_->options.motors) {
    bool found = false;
    for (int attempt = 0; attempt < retries; ++attempt) {
      if (impl_->proto->Ping(cfg.id) == cfg.id) {
        const auto state = impl_->proto->getState();
        if (state != 0) {
          return make_failure(Fault::MotorMissing, cfg.id,
            "motor answered with communication error state: " + std::to_string(state));
        }
        found = true;
        break;
      }
      if (attempt + 1 < retries) {
        std::this_thread::sleep_for(std::chrono::microseconds(retry_gap_us));
      }
    }
    if (!found) {
      return make_failure(Fault::MotorMissing, cfg.id,
        "motor is not responding; check serial chain cables and power supply");
    }
  }
  MotorOpResult ok;
  ok.ok = true;
  return ok;
}

MotorOpResult Bus::apply_motor_config(const MotorConfig & config)
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, config.id, "bus is not open");
  }
  if (impl_->config_of(config.id) == nullptr) {
    return make_failure(
      Fault::WriteRejected, config.id, "motor id is not registered on this bus");
  }
  if (const char * why = validate_motor_config(config)) {
    return make_failure(Fault::WriteRejected, config.id, why);
  }
  std::set<std::uint8_t> unlocked;
  std::vector<std::uint8_t> configured;
  const auto result = impl_->apply_one(config, unlocked, configured);
  auto actual = impl_->config_by_id.at(config.id);
  if (result.ok) {
    actual = config;
  } else {
    // Rollback releases torque and relocks EPROM, but does not undo MODE.
    // Only reconcile this field: the remaining configuration may be partial.
    const int mode = impl_->proto->readByte(config.id, SMS_STS_MODE);
    if ((mode != 0 && mode != 1) || impl_->proto->getState() != 0) {
      return result;
    }
    actual.mode = mode == 1 ? Mode::Wheel : Mode::Position;
  }
  const auto old_mode = impl_->config_by_id.at(config.id).mode;
  if (result.ok || old_mode != actual.mode) {
    if (old_mode != actual.mode) {
      auto & from = old_mode == Mode::Wheel ? impl_->wheel_ids : impl_->position_ids;
      auto & to = actual.mode == Mode::Wheel ? impl_->wheel_ids : impl_->position_ids;
      from.erase(std::remove(from.begin(), from.end(), config.id), from.end());
      to.push_back(config.id);
    }
    impl_->config_by_id.at(config.id) = actual;
    for (auto & registered : impl_->options.motors) {
      if (registered.id == config.id) {
        registered = actual;
        break;
      }
    }
  }
  return result;
}

MotorOpResult Bus::apply_all_configs()
{
  return apply_configs(impl_->all_ids());
}

MotorOpResult Bus::apply_configs(const std::vector<std::uint8_t> & ids)
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  for (const auto id : ids) {
    if (!impl_->config_of(id)) {
      return make_failure(Fault::WriteRejected, id, "motor id is not registered on this bus");
    }
  }
  std::set<std::uint8_t> unlocked;
  std::vector<std::uint8_t> configured;
  for (const auto & cfg : impl_->options.motors) {
    if (std::find(ids.begin(), ids.end(), cfg.id) == ids.end()) {
      continue;
    }
    const MotorOpResult result = impl_->apply_one(cfg, unlocked, configured);
    if (!result.ok) {
      return result;
    }
  }
  MotorOpResult ok;
  ok.ok = true;
  return ok;
}

std::chrono::milliseconds Bus::sync_read_timeout(std::size_t motor_count) const
{
  if (motor_count == 0 || motor_count > impl_->options.motors.size()) {
    throw std::invalid_argument("sync_read_timeout requires a registered motor group size");
  }
  // Match LeRobot's patch_setPacketTimeout: 8N1 wire time, three additional
  // byte times, and 50 ms for USB/host latency. This is a maximum, not a delay.
  const double byte_time_ms = 10'000.0 / impl_->options.baudrate;
  const auto timeout = std::chrono::duration<double, std::milli>(
    byte_time_ms * (motor_count * (kFeedbackLen + 6) + 3) + kSyncReadLatencyMs);
  return std::chrono::ceil<std::chrono::milliseconds>(timeout);
}

MotorOpResult Bus::sync_read(std::vector<MotorSample> & out)
{
  return sync_read(out, impl_->all_ids());
}

MotorOpResult Bus::verify_calibration()
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  for (const auto & cfg : impl_->options.motors) {
    if (cfg.mode != Mode::Position) {
      continue;
    }
    const int offset = impl_->proto->readWord(cfg.id, kRegHomingOfs);
    const int minimum = impl_->proto->readWord(cfg.id, 9);
    const int maximum = impl_->proto->readWord(cfg.id, 11);
    const int mode = impl_->proto->readByte(cfg.id, SMS_STS_MODE);
    if (offset != encode_homing_offset(cfg.homing_offset) || minimum != cfg.range_min ||
        maximum != cfg.range_max || mode != 0 || impl_->proto->getState() != 0)
    {
      return make_failure(Fault::ConfigFailed, cfg.id,
        "firmware calibration mismatch or unreadable motor; provision calibration before read-only connect");
    }
  }
  MotorOpResult ok;
  ok.ok = true;
  return ok;
}

MotorOpResult Bus::sync_read(
  std::vector<MotorSample> & out, const std::vector<std::uint8_t> & ids)
{
  out.clear();
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  if (ids.empty()) {
    return make_failure(Fault::WriteRejected, 0, "sync_read requires at least one motor id");
  }
  if (ids.size() > impl_->options.motors.size()) {
    return make_failure(Fault::WriteRejected, 0, "too many motor ids in sync_read");
  }
  std::set<std::uint8_t> seen;
  for (const std::uint8_t id : ids) {
    if (!seen.insert(id).second) {
      return make_failure(Fault::WriteRejected, id, "duplicate motor id in sync_read");
    }
    if (impl_->config_of(id) == nullptr) {
      return make_failure(
        Fault::WriteRejected, id, "motor id is not registered on this bus");
    }
  }

  // The vendor waits for the session's entire receive capacity, even for subsets.
  if (impl_->sync_read_sized_for != ids.size()) {
    delete[] impl_->proto->syncReadRxBuff;
    impl_->proto->syncReadRxBuff = nullptr;
    impl_->proto->syncReadBegin(
      static_cast<std::uint8_t>(ids.size()), kFeedbackLen,
      static_cast<std::uint32_t>(sync_read_timeout(ids.size()).count()));
    impl_->sync_read_sized_for = ids.size();
  }
  const int received = impl_->proto->syncReadPacketTx(
    const_cast<std::uint8_t *>(ids.data()), static_cast<std::uint8_t>(ids.size()),
    kFeedbackStartAddr, kFeedbackLen);
  if (impl_->real && !impl_->real->write_ok()) {
    impl_->note_failure(Fault::SyncReadFailed);
    return make_failure(Fault::SyncReadFailed, 0, "synchronized read request write incomplete");
  }
  if (received <= 0) {
    impl_->note_failure(Fault::SyncReadFailed);
    return make_failure(Fault::SyncReadFailed, 0, received == 0 ?
      "synchronized read received no response bytes before timeout/deadline" :
      "synchronized read serial receive error");
  }

  constexpr int frame_size = kFeedbackLen + 6;
  auto * const receive_buffer = impl_->proto->syncReadRxBuff;
  auto * const receive_end = receive_buffer + received;
  std::vector<MotorSample> samples;
  samples.reserve(ids.size());
  for (const std::uint8_t id : ids) {
    std::uint8_t data[kFeedbackLen] = {0};
    const std::array<std::uint8_t, 4> header = {0xFF, 0xFF, id, kFeedbackLen + 2};
    auto * const packet = std::search(receive_buffer, receive_end, header.begin(), header.end());
    if (receive_end - packet < frame_size) {
      impl_->note_failure(Fault::MotorMissing);
      return make_failure(
        Fault::MotorMissing, id, "missing or incomplete feedback for motor " + std::to_string(id) +
        " (received " + std::to_string(received) + "/" +
        std::to_string(ids.size() * frame_size) + " bytes)");
    }
    // The vendor checks remaining length before scanning for a header, so a
    // truncated later frame can consume a previous response's unwritten tail.
    // Give its decoder only the complete frame found inside this read's bytes.
    impl_->proto->syncReadRxBuff = packet;
    impl_->proto->syncReadRxBuffLen = frame_size;
    const int decoded = impl_->proto->syncReadPacketRx(id, data);
    impl_->proto->syncReadRxBuff = receive_buffer;
    impl_->proto->syncReadRxBuffLen = static_cast<std::uint16_t>(received);
    if (decoded != kFeedbackLen) {
      impl_->note_failure(Fault::SyncReadFailed);
      return make_failure(Fault::SyncReadFailed, id,
        "feedback checksum mismatch for motor " + std::to_string(id));
    }
    if (impl_->proto->getState() != 0) {
      impl_->note_failure(Fault::SyncReadFailed);
      return make_failure(Fault::SyncReadFailed, id,
        "motor " + std::to_string(id) + " reported feedback status " +
        std::to_string(impl_->proto->getState()));
    }
    const int pos_raw = data[0] | (data[1] << 8);
    const int speed_raw = decode_sign_magnitude15(
      static_cast<std::uint16_t>(data[2] | (data[3] << 8)));
    const int current_raw = decode_sign_magnitude15(
      static_cast<std::uint16_t>(data[13] | (data[14] << 8)));

    MotorSample sample;
    sample.id = id;
    const MotorConfig * cfg = impl_->config_of(id);
    sample.position = (cfg != nullptr && cfg->mode == Mode::Wheel)
                        ? accumulated_ticks_to_radians(pos_raw)
                        : ticks_to_radians(pos_raw);
    sample.velocity = steps_to_rad_per_s(speed_raw);
    sample.effort = raw_current_to_ampere(current_raw);
    sample.valid = true;
    samples.push_back(sample);
  }
  impl_->note_success();
  out = std::move(samples);
  MotorOpResult ok;
  ok.ok = true;
  return ok;
}

MotorOpResult Bus::sync_write_positions(const std::vector<MotorTarget> & targets)
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  if (targets.size() > impl_->options.motors.size()) {
    return make_failure(Fault::WriteRejected, 0, "too many position targets");
  }
  std::set<std::uint8_t> seen;
  std::vector<std::uint8_t> ids;
  std::vector<std::int16_t> positions;
  std::vector<std::uint16_t> speeds;
  std::vector<std::uint8_t> accs;
  std::vector<std::uint8_t> clamped;
  ids.reserve(targets.size());
  positions.reserve(targets.size());
  speeds.reserve(targets.size());
  accs.reserve(targets.size());

  for (const auto & target : targets) {
    if (!std::isfinite(target.position) || !seen.insert(target.id).second) {
      return make_failure(Fault::WriteRejected, target.id, "non-finite or duplicate position target");
    }
    const MotorConfig * cfg = impl_->config_of(target.id);
    if (cfg == nullptr) {
      return make_failure(
        Fault::WriteRejected, target.id, "unknown motor id in position targets");
    }
    if (cfg->mode != Mode::Position) {
      return make_failure(
        Fault::WriteRejected, target.id, "motor is not in position mode");
    }
    if (would_clamp_radians(target.position)) {
      clamped.push_back(target.id);
    }
    ids.push_back(target.id);
    positions.push_back(
      static_cast<std::int16_t>(radians_to_ticks(target.position)));
    speeds.push_back(cfg->profile_speed);
    accs.push_back(cfg->profile_acc);
  }
  if (!ids.empty()) {
    impl_->proto->SyncWritePosEx(
      ids.data(), static_cast<std::uint8_t>(ids.size()), positions.data(), speeds.data(),
      accs.data());
    if (impl_->real && !impl_->real->write_ok()) {
      impl_->note_failure(Fault::WriteRejected);
      return make_failure(Fault::WriteRejected, 0, "position write transport failure");
    }
  }
  MotorOpResult ok;
  ok.ok = true;
  ok.clamped = std::move(clamped);
  return ok;
}

MotorOpResult Bus::sync_write_velocities(const std::vector<MotorTarget> & targets)
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  if (targets.size() > impl_->options.motors.size()) {
    return make_failure(Fault::WriteRejected, 0, "too many velocity targets");
  }
  std::set<std::uint8_t> seen;
  std::vector<std::uint8_t> ids;
  std::vector<std::int16_t> speeds;
  std::vector<std::uint8_t> accs;
  std::vector<std::uint8_t> clamped;
  ids.reserve(targets.size());
  speeds.reserve(targets.size());
  accs.reserve(targets.size());

  for (const auto & target : targets) {
    if (!std::isfinite(target.velocity) || !seen.insert(target.id).second) {
      return make_failure(Fault::WriteRejected, target.id, "non-finite or duplicate velocity target");
    }
    const MotorConfig * cfg = impl_->config_of(target.id);
    if (cfg == nullptr) {
      return make_failure(
        Fault::WriteRejected, target.id, "unknown motor id in velocity targets");
    }
    if (cfg->mode != Mode::Wheel) {
      return make_failure(Fault::WriteRejected, target.id, "motor is not in wheel mode");
    }
    ids.push_back(target.id);
    bool saturated = false;
    speeds.push_back(static_cast<std::int16_t>(rad_per_s_to_steps(target.velocity, &saturated)));
    if (saturated) {
      clamped.push_back(target.id);
    }
    accs.push_back(cfg->profile_acc);
  }
  if (!ids.empty()) {
    impl_->proto->SyncWriteSpe(
      ids.data(), static_cast<std::uint8_t>(ids.size()), speeds.data(), accs.data());
    if (impl_->real && !impl_->real->write_ok()) {
      impl_->note_failure(Fault::WriteRejected);
      return make_failure(Fault::WriteRejected, 0, "velocity write transport failure");
    }
  }
  MotorOpResult ok;
  ok.ok = true;
  ok.clamped = std::move(clamped);
  return ok;
}

MotorOpResult Bus::emergency_release_all()
{
  return emergency_release(impl_->all_ids());
}

MotorOpResult Bus::emergency_release(const std::vector<std::uint8_t> & ids)
{
  if (!impl_->opened) {
    return make_failure(Fault::NotOpen, 0, "bus is not open");
  }
  std::vector<std::uint8_t> unreleased;
  for (const std::uint8_t id : ids) {
    if (impl_->config_of(id) == nullptr) {
      return make_failure(
        Fault::WriteRejected, id, "motor id is not registered on this bus");
    }
    bool disabled = false;
    for (int attempt = 0; attempt < kTorqueRetryCount && !disabled; ++attempt) {
      disabled = impl_->proto->EnableTorque(id, 0) != 0 && impl_->proto->getState() == 0;
    }
    if (!disabled) {
      unreleased.push_back(id);
    }
  }
  if (!unreleased.empty()) {
    std::string detail = "could not release: ";
    for (std::size_t i = 0; i < unreleased.size(); ++i) {
      if (i != 0) {
        detail += ", ";
      }
      detail += std::to_string(unreleased[i]);
    }
    impl_->note_failure(Fault::EmergencyPartiallyFailed);
    return make_failure(Fault::EmergencyPartiallyFailed, unreleased.front(), detail);
  }
  MotorOpResult ok;
  ok.ok = true;
  return ok;
}

const BusHealth & Bus::health() const { return impl_->health; }

const std::vector<MotorConfig> & Bus::motors() const { return impl_->options.motors; }

Bus::SimControl & Bus::sim()
{
  if (!impl_->options.simulated || !impl_->sim_control) {
    throw std::logic_error("Bus::sim() called on a non-simulated bus");
  }
  return *impl_->sim_control;
}

}  // namespace feetech
