# Platform power and battery telemetry published by a robot runtime.
#
# Unknown numeric fields are NaN. Per-rail telemetry is optional and uses
# parallel arrays so a runtime can expose as many rails as it actually reads.

std_msgs/Header header

# Battery
bool battery_present
float32 battery_percentage      # 0..100
float32 battery_voltage         # V
float32 battery_current         # A, positive when charging
float32 battery_temperature     # degC
float32 battery_power           # W
uint32 battery_cycle_count
bool charging

# Platform thermals and fans
float32 temperature             # degC, platform/PMU temperature
float32 fan_speed_rpm
float32 fan_percentage          # 0..100

# Optional per-rail telemetry (parallel arrays, equal length)
string[] rail_names
float32[] rail_voltages         # V
float32[] rail_currents         # A
