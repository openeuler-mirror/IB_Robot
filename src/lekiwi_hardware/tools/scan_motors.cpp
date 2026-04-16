/*
 * Motor Scanner for LeKiwi
 * Scans all possible motor IDs (1-9) and reports which ones are responding
 */

#include <iostream>
#include <unistd.h>
#include "SMS_STS.h"

SMS_STS sms_sts;

int main(int argc, char **argv)
{
    const char* port = "/dev/ttyACM0";
    if (argc >= 2) {
        port = argv[1];
    }

    std::cout << "=== LeKiwi Motor Scanner ===" << std::endl;
    std::cout << "Scanning port: " << port << std::endl;
    std::cout << "Baud rate: 1000000" << std::endl;
    std::cout << std::endl;

    if (!sms_sts.begin(1000000, port)) {
        std::cout << "ERROR: Failed to connect to motor controller on " << port << std::endl;
        std::cout << "Please check:" << std::endl;
        std::cout << "  1. Is the device connected?" << std::endl;
        std::cout << "  2. Do you have permission to access " << port << "?" << std::endl;
        std::cout << "     Try: sudo chmod 666 " << port << std::endl;
        return 1;
    }

    std::cout << "Connected successfully. Scanning for motors..." << std::endl;
    std::cout << std::endl;

    int found_count = 0;
    int expected_ids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int num_expected = 9;

    std::cout << "Scanning expected motor IDs (1-9):" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    for (int i = 0; i < num_expected; i++) {
        int id = expected_ids[i];
        bool found = false;

        // Try 3 times with increasing delays
        for (int retry = 0; retry < 3; retry++) {
            if (sms_sts.Ping(id) != -1) {
                found = true;
                found_count++;
                std::cout << "[✓] Motor ID " << id << " - OK";
                if (id <= 6) {
                    std::cout << " (Arm joint " << id << ")";
                } else {
                    std::cout << " (Base wheel " << (id - 6) << ")";
                }
                std::cout << std::endl;
                break;
            }
            usleep(10000 * (retry + 1)); // 10ms, 20ms, 30ms delays
        }

        if (!found) {
            std::cout << "[✗] Motor ID " << id << " - NOT RESPONDING";
            if (id <= 6) {
                std::cout << " (Arm joint " << id << ")";
            } else {
                std::cout << " (Base wheel " << (id - 6) << ")";
            }
            std::cout << std::endl;
        }
        usleep(50000); // 50ms between motors
    }

    std::cout << "------------------------------------" << std::endl;
    std::cout << "Found " << found_count << "/" << num_expected << " motors" << std::endl;
    std::cout << std::endl;

    if (found_count < num_expected) {
        std::cout << "WARNING: Some motors are not responding!" << std::endl;
        std::cout << std::endl;
        std::cout << "Possible solutions:" << std::endl;
        std::cout << "  1. Check physical connections" << std::endl;
        std::cout << "  2. Check if motors are powered" << std::endl;
        std::cout << "  3. Verify motor IDs match configuration" << std::endl;
        std::cout << "  4. Update URDF motor ID configuration if needed" << std::endl;
        std::cout << "     File: src/lekiwi_description/urdf/lekiwi_car.urdf.xacro" << std::endl;
    } else {
        std::cout << "All motors are responding correctly!" << std::endl;
    }

    sms_sts.end();
    return (found_count == num_expected) ? 0 : 1;
}
