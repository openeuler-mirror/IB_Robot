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

#include "sim_sms_sts.hpp"

#include <algorithm>
#include <cstring>

namespace feetech
{

namespace
{

constexpr std::uint8_t kInstPing = 0x01;
constexpr std::uint8_t kInstRead = 0x02;
constexpr std::uint8_t kInstWrite = 0x03;
constexpr std::uint8_t kInstSyncRead = 0x82;
constexpr std::uint8_t kInstSyncWrite = 0x83;

// Register addresses (mirror SMS_STS.h).
constexpr std::uint8_t kRegEpromFirst = 5;
constexpr std::uint8_t kRegEpromLast = 33;
constexpr std::uint8_t kRegMode = 33;
constexpr std::uint8_t kRegTorqueEnable = 40;
constexpr std::uint8_t kRegAcc = 41;
constexpr std::uint8_t kRegLock = 55;

std::uint16_t word_le(const std::uint8_t * p)
{
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint16_t encode_word(int value)
{
  // Sign-magnitude with bit 15 as the sign flag.
  if (value < 0) {
    return static_cast<std::uint16_t>(-value) | 0x8000U;
  }
  return static_cast<std::uint16_t>(value);
}

int decode_word_signed(std::uint16_t word)
{
  if (word & 0x8000U) {
    return -static_cast<int>(word & 0x7FFFU);
  }
  return static_cast<int>(word);
}

}  // namespace

SimSmsSts::SimSmsSts()
{
  // The vendor default constructor leaves End=0 (little endian words on the
  // wire), matching SCSerial/SO-101 usage. Level=1 means ack'd writes.
}

void SimSmsSts::add_motor(std::uint8_t id)
{
  Motor initial;
  initial.regs[11] = 0xFF;
  initial.regs[12] = 0x0F;
  motors_.emplace(id, std::move(initial));
}

SimSmsSts::Motor * SimSmsSts::motor(std::uint8_t id)
{
  auto it = motors_.find(id);
  return it == motors_.end() ? nullptr : &it->second;
}

void SimSmsSts::queue_bytes(const std::vector<std::uint8_t> & bytes)
{
  rx_.insert(rx_.end(), bytes.begin(), bytes.end());
}

void SimSmsSts::queue_ack(std::uint8_t id)
{
  // FF FF ID 02 STATUS CHECKSUM, checksum = ~(ID + 2 + STATUS)
  const std::uint8_t status = motor(id)->response_status;
  const std::uint8_t checksum =
    static_cast<std::uint8_t>(~(id + 2 + status) & 0xFF);
  queue_bytes({0xFF, 0xFF, id, 2, status, checksum});
}

void SimSmsSts::advance_time()
{
  // Called once per read round: position-mode motors step toward their goal,
  // wheel-mode motors accumulate rotation at the commanded speed.
  for (auto & [id, m] : motors_) {
    if (!m.present || !m.torque_on) {
      m.last_move_ticks = 0;
      continue;
    }
    if (m.wheel_mode) {
      m.pos_ticks += m.wheel_speed_steps;
      m.last_move_ticks = m.wheel_speed_steps;
    } else {
      const std::int32_t delta = m.goal_ticks - m.pos_ticks;
      const std::int32_t step = std::clamp(delta, -converge_step_ticks, converge_step_ticks);
      m.pos_ticks += step;
      m.last_move_ticks = step;
    }
    // Wrap into the 12-bit encoder range like the real servo.
    m.pos_ticks = ((m.pos_ticks % 4096) + 4096) % 4096;
  }
}

std::vector<std::uint8_t> SimSmsSts::register_block(
  std::uint8_t id, std::uint8_t addr, int len) const
{
  auto it = motors_.find(id);
  std::vector<std::uint8_t> block(static_cast<std::size_t>(len), 0);
  if (it == motors_.end()) {
    return block;
  }
  const Motor & m = it->second;
  for (int i = 0; i < len; ++i) {
    const std::uint16_t reg = static_cast<std::uint16_t>(addr) + static_cast<std::uint16_t>(i);
    std::uint8_t value = 0;
    switch (reg) {
      case kRegMode: value = m.wheel_mode ? 1 : 0; break;
      case kRegTorqueEnable: value = m.torque_on ? 1 : 0; break;
      case kRegLock: value = m.eprom_locked ? 1 : 0; break;
      case 56: value = static_cast<std::uint8_t>(m.pos_ticks & 0xFF); break;
      case 57: value = static_cast<std::uint8_t>((m.pos_ticks >> 8) & 0xFF); break;
      case 58: {
        const std::uint16_t speed = encode_word(m.last_move_ticks);
        value = static_cast<std::uint8_t>(speed & 0xFF);
        break;
      }
      case 59: {
        const std::uint16_t speed = encode_word(m.last_move_ticks);
        value = static_cast<std::uint8_t>((speed >> 8) & 0xFF);
        break;
      }
      default: {
        auto reg_it = m.regs.find(static_cast<std::uint8_t>(reg));
        if (reg_it != m.regs.end()) {
          value = reg_it->second;
        }
        break;
      }
    }
    block[static_cast<std::size_t>(i)] = value;
  }
  return block;
}

bool SimSmsSts::write_register(
  std::uint8_t id, std::uint8_t addr, const std::uint8_t * data, int len)
{
  Motor * m = motor(id);
  if (m == nullptr || !m->present) {
    return false;
  }
  // Ack gating: injected failures, register-precise rejections, and (for
  // EPROM registers) the lock state.
  if (!m->acks_writes) {
    return false;
  }
  if (m->fail_next_writes > 0) {
    --m->fail_next_writes;
    return false;
  }
  const auto rejected = m->rejected.find(addr);
  if (rejected != m->rejected.end() &&
      (rejected->second.empty() || rejected->second.count(data[0]) != 0))
  {
    return false;
  }
  const bool is_eprom = addr >= kRegEpromFirst && addr <= kRegEpromLast;
  if (is_eprom && m->eprom_locked) {
    return false;
  }

  if (addr == kRegLock) {
    m->eprom_locked = data[0] != 0;
  } else if (addr == kRegTorqueEnable) {
    m->torque_on = (data[0] & 1) != 0;
  } else if (addr == kRegMode) {
    m->wheel_mode = data[0] == 1;
  } else if (addr == kRegAcc && len == 7) {
    // Position/velocity command frame: [acc, posL, posH, timeL, timeH,
    // speedL, speedH]. Position is sign-magnitude; the wheel velocity word
    // is sign-magnitude at bytes 5-6.
    const int goal = decode_word_signed(word_le(data + 1));
    const int speed = decode_word_signed(word_le(data + 5));
    if (m->wheel_mode) {
      m->wheel_speed_steps = speed;
    } else {
      m->goal_ticks = goal;
    }
  } else {
    for (int i = 0; i < len; ++i) {
      m->regs[static_cast<std::uint8_t>(addr + i)] = data[i];
    }
  }
  return true;
}

void SimSmsSts::on_frame(const std::vector<std::uint8_t> & frame)
{
  // frame: FF FF ID LEN FUNC PARAMS... CHECKSUM (LEN covers FUNC..CHECKSUM).
  const std::uint8_t id = frame[2];
  const std::uint8_t len = frame[3];
  const std::uint8_t func = frame[4];
  const std::uint8_t * params = frame.data() + 5;
  const int params_len = static_cast<int>(len) - 2;  // FUNC + CHECKSUM excluded

  switch (func) {
    case kInstPing: {
      if (id != 0xFE && motor(id) != nullptr && motor(id)->present && motor(id)->responsive) {
        queue_ack(id);
      }
      break;
    }
    case kInstWrite: {
      if (params_len < 1) {
        break;
      }
      const std::uint8_t addr = params[0];
      const std::uint8_t * data = params + 1;
      const int data_len = params_len - 1;
      const bool applied = write_register(id, addr, data, data_len);
      if (applied && id != 0xFE && Level) {
        queue_ack(id);
      }
      break;
    }
    case kInstRead: {
      if (params_len < 2 || id == 0xFE) {
        break;
      }
      Motor * m = motor(id);
      if (m == nullptr || !m->present || !m->responsive) {
        break;
      }
      advance_time();
      const std::uint8_t addr = params[0];
      const int read_len = params[1];
      const auto block = register_block(id, addr, read_len);
      // Response: FF FF ID (read_len+2) STATUS DATA CHECKSUM.
      std::vector<std::uint8_t> response{0xFF, 0xFF, id,
        static_cast<std::uint8_t>(read_len + 2), m->response_status};
      std::uint16_t sum = id + (read_len + 2) + m->response_status;
      for (std::uint8_t byte : block) {
        response.push_back(byte);
        sum += byte;
      }
      response.push_back(static_cast<std::uint8_t>(~sum & 0xFF));
      queue_bytes(response);
      break;
    }
    case kInstSyncWrite: {
      // Broadcast frame: FF FF FE LEN 0x83 ADDR nLen [ID DATA...]...
      if (params_len < 2) {
        break;
      }
      const std::uint8_t addr = params[0];
      const int n_len = params[1];
      const std::uint8_t * p = params + 2;
      const std::uint8_t * end = params + params_len;
      while (p + 1 + n_len <= end) {
        const std::uint8_t target = p[0];
        write_register(target, addr, p + 1, n_len);
        p += 1 + n_len;
      }
      break;
    }
    case kInstSyncRead: {
      // Broadcast frame: FF FF FE LEN 0x82 ADDR nLen ID...
      if (params_len < 3 || drop_next_sync_read) {
        drop_next_sync_read = false;
        break;
      }
      const std::uint8_t addr = params[0];
      const int read_len = params[1];
      const std::uint8_t * ids = params + 2;
      const int id_count = params_len - 2;
      advance_time();
      for (int i = 0; i < id_count; ++i) {
        const std::uint8_t target = ids[i];
        Motor * m = motor(target);
        if (m == nullptr || !m->present || !m->responsive) {
          continue;  // unresponsive motors contribute no packet
        }
        const auto block = register_block(target, addr, read_len);
        std::vector<std::uint8_t> response{0xFF, 0xFF, target,
          static_cast<std::uint8_t>(read_len + 2), m->response_status};
        std::uint16_t sum = target + (read_len + 2) + m->response_status;
        for (std::uint8_t byte : block) {
          response.push_back(byte);
          sum += byte;
        }
        response.push_back(static_cast<std::uint8_t>(~sum & 0xFF));
        queue_bytes(response);
      }
      break;
    }
    default:
      break;  // REG_WRITE/ACTION/RESET/CAL are not used by this SDK
  }
}

int SimSmsSts::writeSCS(unsigned char * nDat, int nLen)
{
  for (int i = 0; i < nLen; ++i) {
    const std::uint8_t byte = nDat[i];
    if (frame_.empty()) {
      if (byte == 0xFF) {
        frame_.push_back(0xFF);
      }
      continue;
    }
    if (frame_.size() == 1) {
      if (byte == 0xFF) {
        frame_.push_back(0xFF);
      } else {
        frame_.clear();
      }
      continue;
    }
    frame_.push_back(byte);  // id, len, func, params, checksum
    if (frame_.size() >= 4) {
      const std::size_t total = 4 + frame_[3];
      if (frame_.size() == total) {
        on_frame(frame_);
        frame_.clear();
      }
    }
  }
  return nLen;
}

int SimSmsSts::writeSCS(unsigned char bDat)
{
  return writeSCS(&bDat, 1);
}

int SimSmsSts::readSCS(unsigned char * nDat, int nLen)
{
  if (nDat == nullptr || nLen <= 0) {
    return 0;
  }
  const std::size_t count = std::min<std::size_t>(rx_.size(), static_cast<std::size_t>(nLen));
  std::memcpy(nDat, rx_.data(), count);
  rx_.erase(rx_.begin(), rx_.begin() + static_cast<std::ptrdiff_t>(count));
  return static_cast<int>(count);
}

int SimSmsSts::readSCS(unsigned char * nDat, int nLen, unsigned long /*TimeOut*/)
{
  return readSCS(nDat, nLen);
}

void SimSmsSts::rFlushSCS()
{
  rx_.clear();
}

void SimSmsSts::wFlushSCS() {}

}  // namespace feetech
