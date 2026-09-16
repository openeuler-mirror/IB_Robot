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

// Exercise serial transport behavior with pipe descriptors, including the
// nonblocking backpressure used by the vendor's serial descriptor.

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <pthread.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "feetech/bus.hpp"
#include "safe_sms_sts.hpp"
#include "sim_sms_sts.hpp"

namespace
{

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

TEST(SerialHardening, SimulatedStatusIsParsedByVendorProtocol)
{
  feetech::SimSmsSts servo;
  servo.add_motor(1);
  servo.add_motor(2);
  servo.motor(1)->response_status = 0x20;
  EXPECT_EQ(servo.Ping(1), 1);
  EXPECT_EQ(servo.getState(), 0x20);
  EXPECT_EQ(servo.EnableTorque(1, 0), 1);
  EXPECT_EQ(servo.getState(), 0x20);
  EXPECT_EQ(servo.readByte(1, SMS_STS_MODE), 0);
  EXPECT_EQ(servo.getState(), 0x20);
  std::uint8_t ids[] = {1, 2};
  std::uint8_t data[1] = {};
  servo.syncReadBegin(2, 1, 20);
  EXPECT_GT(servo.syncReadPacketTx(ids, 2, SMS_STS_MODE, 1), 0);
  EXPECT_EQ(servo.syncReadPacketRx(1, data), 1);
  EXPECT_EQ(servo.getState(), 0x20);
  EXPECT_EQ(servo.syncReadPacketRx(2, data), 1);
  EXPECT_EQ(servo.getState(), 0);
  delete[] servo.syncReadRxBuff;
  servo.syncReadRxBuff = nullptr;
}

class TestableSafeSmsSts : public feetech::SafeSmsSts
{
public:
  int read_for_test(unsigned char * data, int length, unsigned long timeout_ms)
  {
    return readSCS(data, length, timeout_ms);
  }
  void set_fd_for_test(int value) { fd = value; }
  void set_timeout_for_test(unsigned long timeout_ms) { IOTimeOut = timeout_ms; }
  void write_for_test()
  {
    writeSCS(static_cast<unsigned char>(42));
    wFlushSCS();
  }
  void write_for_test(unsigned char * data, int length)
  {
    writeSCS(data, length);
    wFlushSCS();
  }
};

class NonblockingSerial : public ::testing::Test
{
protected:
  void SetUp() override
  {
    ASSERT_EQ(pipe2(descriptors, O_NONBLOCK), 0);
    servo.set_timeout_for_test(1000);
  }

  void TearDown() override
  {
    servo.set_fd_for_test(-1);
    for (const int descriptor : descriptors) {
      if (descriptor >= 0) {
        close(descriptor);
      }
    }
  }

  void fill_pipe()
  {
    unsigned char bytes[4096]{};
    ssize_t count;
    while ((count = ::write(descriptors[1], bytes, sizeof(bytes))) > 0) {
      filled += count;
    }
    ASSERT_EQ(count, -1);
    ASSERT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
    ASSERT_GT(filled, 0);
  }

  void drain_filler()
  {
    unsigned char bytes[4096];
    while (filled > 0) {
      const auto count = ::read(descriptors[0], bytes, sizeof(bytes));
      ASSERT_GT(count, 0);
      filled -= count;
    }
  }

  TestableSafeSmsSts servo;
  int descriptors[2] = {-1, -1};
  ssize_t filled = 0;
};

volatile std::sig_atomic_t interruptions = 0;

void handle_interrupt(int)
{
  ++interruptions;
}

// Interrupt only the calling test thread, without SA_RESTART. Keep sending
// beyond the deadline so restarting poll with a fresh timeout is observable.
class InterruptPoll
{
public:
  InterruptPoll()
  {
    interruptions = 0;
    struct sigaction action{};
    action.sa_handler = handle_interrupt;
    sigemptyset(&action.sa_mask);
    EXPECT_EQ(sigaction(SIGUSR1, &action, &previous_), 0);
    const auto target = pthread_self();
    thread_ = std::thread([this, target]() {
        const auto until = Clock::now() + 300ms;
        while (!stop_ && Clock::now() < until) {
          pthread_kill(target, SIGUSR1);
          std::this_thread::sleep_for(1ms);
        }
      });
  }

  ~InterruptPoll()
  {
    stop_ = true;
    thread_.join();
    EXPECT_EQ(sigaction(SIGUSR1, &previous_, nullptr), 0);
  }

private:
  struct sigaction previous_{};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

TEST(SerialHardening, InvalidDescriptorFailsWithoutTouchingBuffer)
{
  TestableSafeSmsSts servo;
  unsigned char buffer[2] = {0xAA, 0xBB};
  servo.set_fd_for_test(-1);

  EXPECT_EQ(servo.read_for_test(buffer, 1, 1), -1);
  EXPECT_EQ(buffer[0], 0xAA);
  EXPECT_EQ(buffer[1], 0xBB);
}

TEST(SerialHardening, ClosedPeerYieldsZeroNotNegative)
{
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  close(descriptors[1]);
  TestableSafeSmsSts servo;
  servo.set_fd_for_test(descriptors[0]);
  unsigned char buffer[2] = {0xAA, 0xBB};

  EXPECT_EQ(servo.read_for_test(buffer, 1, 10), 0);
  EXPECT_EQ(buffer[0], 0xAA);
  EXPECT_EQ(buffer[1], 0xBB);
  close(descriptors[0]);
  servo.set_fd_for_test(-1);
}

TEST(SerialHardening, EndActuallyClosesDescriptorAndIsIdempotent)
{
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  TestableSafeSmsSts servo;
  servo.set_fd_for_test(descriptors[0]);
  servo.end();
  EXPECT_EQ(fcntl(descriptors[0], F_GETFD), -1);
  servo.end();
  close(descriptors[1]);
}

TEST(SerialHardening, FailedFlushIsObservable)
{
  TestableSafeSmsSts servo;
  servo.write_for_test();
  EXPECT_FALSE(servo.write_ok());
}

TEST_F(NonblockingSerial, BackpressureWaitsForWritableAndPreservesBytes)
{
  servo.set_fd_for_test(descriptors[1]);
  fill_pipe();
  std::array<unsigned char, 200> expected{};
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<unsigned char>(i);
  }
  std::thread consumer([this]() {
      std::this_thread::sleep_for(20ms);
      drain_filler();
    });
  servo.write_for_test(expected.data(), static_cast<int>(expected.size()));
  consumer.join();

  EXPECT_TRUE(servo.write_ok());
  std::array<unsigned char, 200> actual{};
  ASSERT_EQ(::read(descriptors[0], actual.data(), actual.size()),
    static_cast<ssize_t>(actual.size()));
  EXPECT_EQ(actual, expected);
  unsigned char byte = 0;
  EXPECT_EQ(::read(descriptors[0], &byte, 1), -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
}

TEST_F(NonblockingSerial, WriteTimeoutBoundsBackpressureDespiteInterrupts)
{
  servo.set_fd_for_test(descriptors[1]);
  servo.set_timeout_for_test(40);
  servo.set_io_deadline(Clock::now() + 1s);
  fill_pipe();
  InterruptPoll interrupts;
  const auto start = Clock::now();
  servo.write_for_test();
  const auto elapsed = Clock::now() - start;

  EXPECT_FALSE(servo.write_ok());
  EXPECT_GE(elapsed, 30ms);
  EXPECT_LT(elapsed, 200ms);
  EXPECT_GT(interruptions, 0);
}

TEST_F(NonblockingSerial, CycleDeadlineBoundsBackpressureDespiteInterrupts)
{
  servo.set_fd_for_test(descriptors[1]);
  fill_pipe();
  InterruptPoll interrupts;
  const auto start = Clock::now();
  servo.set_io_deadline(start + 40ms);
  servo.write_for_test();
  const auto elapsed = Clock::now() - start;

  EXPECT_FALSE(servo.write_ok());
  EXPECT_GE(elapsed, 30ms);
  EXPECT_LT(elapsed, 200ms);
  EXPECT_GT(interruptions, 0);
}

TEST_F(NonblockingSerial, CycleDeadlineBoundsReadDespiteInterrupts)
{
  servo.set_fd_for_test(descriptors[0]);
  unsigned char byte = 0xAA;
  InterruptPoll interrupts;
  const auto start = Clock::now();
  servo.set_io_deadline(start + 40ms);
  EXPECT_EQ(servo.read_for_test(&byte, 1, 1000), 0);
  const auto elapsed = Clock::now() - start;

  EXPECT_EQ(byte, 0xAA);
  EXPECT_GE(elapsed, 30ms);
  EXPECT_LT(elapsed, 200ms);
  EXPECT_GT(interruptions, 0);
}

TEST_F(NonblockingSerial, ReadTimeoutIsNotExtendedByCycleDeadline)
{
  servo.set_fd_for_test(descriptors[0]);
  servo.set_io_deadline(Clock::now() + 1s);
  unsigned char byte = 0xAA;
  const auto start = Clock::now();
  EXPECT_EQ(servo.read_for_test(&byte, 1, 40), 0);
  const auto elapsed = Clock::now() - start;

  EXPECT_GE(elapsed, 30ms);
  EXPECT_LT(elapsed, 200ms);
  EXPECT_EQ(byte, 0xAA);
}

TEST_F(NonblockingSerial, ExpiredDeadlinePersistsAcrossReadAndWriteUntilCleared)
{
  const unsigned char expected = 7;
  ASSERT_EQ(::write(descriptors[1], &expected, 1), 1);
  servo.set_io_deadline(Clock::now() - 1ms);
  servo.set_fd_for_test(descriptors[0]);
  unsigned char byte = 0xAA;
  EXPECT_EQ(servo.read_for_test(&byte, 1, 1000), 0);
  EXPECT_EQ(byte, 0xAA);
  servo.set_fd_for_test(descriptors[1]);
  servo.write_for_test();
  EXPECT_FALSE(servo.write_ok());

  servo.clear_io_deadline();
  servo.set_fd_for_test(descriptors[0]);
  ASSERT_EQ(servo.read_for_test(&byte, 1, 1000), 1);
  EXPECT_EQ(byte, expected);
  servo.set_fd_for_test(descriptors[1]);
  servo.write_for_test();
  EXPECT_TRUE(servo.write_ok());
  ASSERT_EQ(::read(descriptors[0], &byte, 1), 1);
  EXPECT_EQ(byte, 42);
  // A failed flush must not leave stale bytes queued for the next frame.
  EXPECT_EQ(::read(descriptors[0], &byte, 1), -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
}

TEST_F(NonblockingSerial, ReadDeadlineReturnsPartialData)
{
  servo.set_fd_for_test(descriptors[0]);
  const unsigned char expected = 7;
  ASSERT_EQ(::write(descriptors[1], &expected, 1), 1);
  servo.set_io_deadline(Clock::now() + 40ms);
  unsigned char bytes[2] = {0xAA, 0xBB};
  EXPECT_EQ(servo.read_for_test(bytes, 2, 1000), 1);
  EXPECT_EQ(bytes[0], expected);
  EXPECT_EQ(bytes[1], 0xBB);
}

class PtySerial : public ::testing::Test
{
protected:
  void SetUp() override
  {
    master = posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
    ASSERT_GE(master, 0);
    ASSERT_EQ(grantpt(master), 0);
    ASSERT_EQ(unlockpt(master), 0);
    const char * name = ptsname(master);
    ASSERT_NE(name, nullptr);
    port = name;
  }

  void TearDown() override
  {
    if (master >= 0) {
      close(master);
    }
  }

  // Read exactly one request, tolerating serial packet fragmentation.
  bool read_request_bytes(unsigned char * bytes, size_t length)
  {
    const auto deadline = Clock::now() + 1s;
    size_t received = 0;
    while (received < length && Clock::now() < deadline) {
      pollfd descriptor{master, POLLIN, 0};
      const int ready = poll(&descriptor, 1, 10);
      if (ready < 0 && errno != EINTR) {
        return false;
      }
      if (ready <= 0) {
        continue;
      }
      const auto count = ::read(master, bytes + received, length - received);
      if (count > 0) {
        received += static_cast<size_t>(count);
      } else if (count == 0 || (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) {
        return false;
      }
    }
    return received == length;
  }

  void answer_sync_read(
    const std::vector<std::uint8_t> & expected_ids, size_t truncate_bytes = 0,
    bool corrupt_checksum = false, std::chrono::milliseconds response_delay = 0ms)
  {
    std::array<unsigned char, 4> header{};
    ASSERT_TRUE(read_request_bytes(header.data(), header.size()));
    ASSERT_EQ(header[0], 0xFF);
    ASSERT_EQ(header[1], 0xFF);
    ASSERT_EQ(header[2], 0xFE);
    ASSERT_GE(header[3], 5);
    std::vector<unsigned char> body(header[3]);
    ASSERT_TRUE(read_request_bytes(body.data(), body.size()));
    ASSERT_EQ(body[0], 0x82);  // SYNC_READ
    ASSERT_EQ(body[1], 56);  // Present position through present current.
    ASSERT_EQ(body[2], 15);
    const std::vector<std::uint8_t> requested_ids(body.begin() + 3, body.end() - 1);
    EXPECT_EQ(requested_ids, expected_ids);

    // Only requested motors respond. Extra registered motors stay silent.
    std::vector<unsigned char> replies;
    for (const auto id : requested_ids) {
      std::vector<unsigned char> reply = {0xFF, 0xFF, id, 17, 0};
      reply.resize(20, 0);
      reply[6] = 8;  // Position = 2048 ticks.
      unsigned char checksum = 0;
      for (size_t i = 2; i < reply.size(); ++i) {
        checksum += reply[i];
      }
      reply.push_back(static_cast<unsigned char>(~checksum));
      replies.insert(replies.end(), reply.begin(), reply.end());
    }
    if (corrupt_checksum) {
      replies.back() ^= 1;
    }
    ASSERT_LT(truncate_bytes, replies.size());
    replies.resize(replies.size() - truncate_bytes);
    std::this_thread::sleep_for(response_delay);
    ASSERT_EQ(::write(master, replies.data(), replies.size()),
      static_cast<ssize_t>(replies.size()));
  }

  int master = -1;
  std::string port;
};

TEST_F(PtySerial, SubsetSyncReadDoesNotWaitForUnrequestedMotorsAndFullReadStillWorks)
{
  feetech::BusOptions options;
  options.port = port;
  for (const auto id : {1, 2, 3}) {
    feetech::MotorConfig motor;
    motor.id = static_cast<std::uint8_t>(id);
    options.motors.push_back(motor);
  }
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  // Alternate shrinking and growing the receive capacity. Use the fastest of
  // three subset rounds so an isolated scheduler delay does not fail the test.
  auto fastest_subset = Clock::duration::max();
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (const auto & ids : std::vector<std::vector<std::uint8_t>>{{2}, {1, 2, 3}}) {
      std::thread responder([this, &ids]() {answer_sync_read(ids);});
      std::vector<feetech::MotorSample> samples;
      const auto start = Clock::now();
      const auto result = ids.size() == 3 ? bus.sync_read(samples) : bus.sync_read(samples, ids);
      const auto elapsed = Clock::now() - start;
      responder.join();
      ASSERT_TRUE(result.ok) << result.detail;
      ASSERT_EQ(samples.size(), ids.size());
      for (size_t i = 0; i < ids.size(); ++i) {
        EXPECT_EQ(samples[i].id, ids[i]);
        EXPECT_TRUE(samples[i].valid);
      }
      if (ids.size() == 1 && elapsed < fastest_subset) {
        fastest_subset = elapsed;
      }
    }
  }
  // A healthy subset returns immediately, without waiting for the timeout.
  EXPECT_LT(fastest_subset, 15ms);
}

TEST_F(PtySerial, DelayedFeedbackUsesLerobotLatencyAllowance)
{
  feetech::BusOptions options;
  options.port = port;
  for (std::uint8_t id = 1; id <= 6; ++id) {
    feetech::MotorConfig motor;
    motor.id = id;
    options.motors.push_back(motor);
  }
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  // Cover both the initial full-bus session and a resized subset session.
  for (const auto & ids : std::vector<std::vector<std::uint8_t>>{{1, 2, 3, 4, 5, 6}, {2}}) {
    std::thread responder([this, &ids]() {answer_sync_read(ids, 0, false, 30ms);});
    std::vector<feetech::MotorSample> samples;
    const auto result = bus.sync_read(samples, ids);
    responder.join();
    ASSERT_TRUE(result.ok) << result.detail;
    ASSERT_EQ(samples.size(), ids.size());
  }

  const auto started = Clock::now();
  std::vector<feetech::MotorSample> samples;
  EXPECT_FALSE(bus.sync_read(samples).ok);
  const auto elapsed = Clock::now() - started;
  EXPECT_GE(elapsed, 50ms);
  EXPECT_LT(elapsed, 150ms);
  EXPECT_TRUE(samples.empty());
}

TEST_F(PtySerial, BusCycleDeadlineBoundsMissingFeedbackAndExpiredWrites)
{
  feetech::BusOptions options;
  options.port = port;
  feetech::MotorConfig motor;
  motor.id = 1;
  options.motors.push_back(motor);
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());
  auto fastest = Clock::duration::max();
  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto started = Clock::now();
    bus.set_io_deadline(started + 4ms);
    std::vector<feetech::MotorSample> samples;
    EXPECT_FALSE(bus.sync_read(samples).ok);
    const auto elapsed = Clock::now() - started;
    EXPECT_GE(elapsed, 4ms);
    if (elapsed < fastest) {
      fastest = elapsed;
    }
    feetech::MotorTarget target;
    target.id = 1;
    EXPECT_FALSE(bus.sync_write_positions({target}).ok);
    bus.clear_io_deadline();
    EXPECT_TRUE(bus.sync_write_positions({target}).ok);
  }
  EXPECT_LT(fastest, 10ms);
}

TEST_F(PtySerial, TruncatedFeedbackCannotReusePreviousResponseTail)
{
  feetech::BusOptions options;
  options.port = port;
  for (const auto id : {1, 2}) {
    feetech::MotorConfig motor;
    motor.id = static_cast<std::uint8_t>(id);
    options.motors.push_back(motor);
  }
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  // Reuse the same receive buffer: its unwritten tail still contains the
  // preceding valid packet, including its checksum.
  for (const size_t truncate_bytes : {0U, 5U}) {
    std::thread responder([this, truncate_bytes]() {
        answer_sync_read({1, 2}, truncate_bytes);
      });
    std::vector<feetech::MotorSample> samples;
    const auto result = bus.sync_read(samples);
    responder.join();
    if (truncate_bytes == 0) {
      ASSERT_TRUE(result.ok) << result.detail;
      ASSERT_EQ(samples.size(), 2U);
    } else {
      EXPECT_FALSE(result.ok);
      EXPECT_EQ(result.failed_id, 2U);
      EXPECT_EQ(result.fault, feetech::Fault::MotorMissing);
      EXPECT_TRUE(samples.empty());
      EXPECT_NE(result.detail.find("motor 2"), std::string::npos);
      EXPECT_NE(result.detail.find("37/42 bytes"), std::string::npos);
    }
  }
}

TEST_F(PtySerial, CorruptFeedbackIsNotReportedAsAnAbsentMotor)
{
  feetech::BusOptions options;
  options.port = port;
  feetech::MotorConfig motor;
  motor.id = 1;
  options.motors.push_back(motor);
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  std::thread responder([this]() {answer_sync_read({1}, 0, true);});
  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  responder.join();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::SyncReadFailed);
  EXPECT_EQ(result.failed_id, 1U);
  EXPECT_TRUE(samples.empty());
  EXPECT_NE(result.detail.find("checksum mismatch for motor 1"), std::string::npos);
}

TEST_F(PtySerial, ExpiredReadRequestIsReportedAsAnIncompleteWrite)
{
  feetech::BusOptions options;
  options.port = port;
  feetech::MotorConfig motor;
  motor.id = 1;
  options.motors.push_back(motor);
  feetech::Bus bus(options);
  ASSERT_TRUE(bus.open());

  bus.set_io_deadline(Clock::now() - 1ms);
  std::vector<feetech::MotorSample> samples;
  const auto result = bus.sync_read(samples);
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.fault, feetech::Fault::SyncReadFailed);
  EXPECT_TRUE(samples.empty());
  EXPECT_NE(result.detail.find("request write incomplete"), std::string::npos);
}

}  // namespace
