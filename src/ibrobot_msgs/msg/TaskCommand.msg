string task_id
string source          # 请求来源：voice, text, vision, api
string raw_command
string task_type       # 任务类型：pick, place, move, push, pull 等
string target_name
string place_name
string motion_direction  # 运动方向：forward, backward, left, right, up, down
float32 motion_distance
uint8 priority         # 优先级：0=低, 1=中, 2=高, 3=紧急
float32 timeout_sec
string context_json
