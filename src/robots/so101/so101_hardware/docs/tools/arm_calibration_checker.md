# arm_calibration_checker

SO-101 follower 机械臂校准检查工具。它会按固定顺序驱动机械臂到一组预设姿态，帮助你逐项确认关节角度、腕部翻转和夹爪开合是否与预期一致。

## 适用场景

- 新机械臂刚完成标定，想快速确认标定是否可信
- 机械臂经历拆装、碰撞或更换舵机后，怀疑零位或装配出现漂移
- 已有标定文件，但需要再做一轮人工量具检查

## 工具能力

当前实现优先保持原始迁移工具的调用方式，面向真机 follower 检查流程。

## 默认检查步骤

工具会按如下顺序提示并执行：

1. 30° 测量动作
2. 60° 测量动作
3. 90° 测量动作
4. 回到 0 位
5. `wrist_roll` 转到“镜头在正下方”
6. `wrist_roll` 转到“镜头在正上方”
7. 夹爪完全打开
8. 回到初始位置
9. 完成收臂动作

## ROS 2 环境准备

推荐在 IB_Robot 仓根目录执行：

```bash
cd /path/to/IB_Robot
source .shrc_local
```

如果 `so101_hardware` 尚未编译，先执行：

```bash
cd /path/to/IB_Robot
source .shrc_local
colcon build --merge-install --symlink-install --packages-select so101_hardware
```

## ROS 2 中的使用方式

最常见的调用方式如下：

```bash
cd /path/to/IB_Robot
source .shrc_local

ros2 run so101_hardware arm_calibration_checker \
  --port /dev/ttyACM0 \
  --calib-file ~/.calibrate/so101_follower_calibrate.json
```

兼容旧参数风格的写法也保留了，例如：

```bash
ros2 run so101_hardware arm_calibration_checker \
  --robot.type so101_follower \
  --robot.port /dev/ttyACM0 \
  --robot.id single_follower_arm
```

其中：

- `--robot.type` 目前只接受 `so101_follower`
- `--robot.id` 会优先尝试按旧 LeRobot 目录规则定位标定文件
- 旧 LeRobot 默认标定目录形如 `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot.id>.json`
- 若显式传入 `--robot.calibration_dir`，则会按 `<robot.calibration_dir>/<robot.id>.json` 解析
- 若以上旧路径都不可用，才会回退到当前 IB_Robot 默认文件 `~/.calibrate/so101_follower_calibrate.json`

如果你希望显式指定旧版标定目录，也可以这样写：

```bash
ros2 run so101_hardware arm_calibration_checker \
  --robot.type so101_follower \
  --robot.port /dev/ttyACM0 \
  --robot.id single_follower_arm \
  --robot.calibration_dir ~/.cache/huggingface/lerobot/calibration/robots/so101_follower
```

工具当前同时兼容两类 follower 标定文件：

- IB_Robot 当前格式：关节键为 `"1"` 到 `"6"`
- 旧 LeRobot 格式：关节键为 `shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_roll`、`gripper`

## 建议的检查方式

### 关节 1-4

建议准备 30° / 60° / 90° 三角尺，观察：

- 关节是否到达提示对应角度
- 连杆与桌面或上一级连杆夹角是否明显偏差
- 不同步骤之间是否有系统性偏大或偏小

### 关节 5

重点看 `wrist_roll` 翻到正上方 / 正下方时：

- 相机或末端是否明显倾斜
- 旋转过程中是否有干涉
- 视觉中心是否出现大幅“画圈”

### 关节 6

重点看夹爪完全打开时：

- 是否确实达到最大开口
- 两侧开合同步是否正常
- 舵机是否出现堵转异响

## 与其他工具的关系

这个工具最适合放在以下流程里使用：

1. 先用 `calibrate_arm` 生成基础标定
2. 如需复用旧标定逻辑，再用 `arm_calibration_transfer`
3. 最后用 `arm_calibration_checker` 做人工核验

## 常见问题

### 1. 真机模式提示找不到标定文件

当前仓默认 follower 标定文件通常在：

```text
~/.calibrate/so101_follower_calibrate.json
```

旧用法下，如果你只传了 `--robot.id`，工具会先尝试：

```text
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<robot.id>.json
```

如果你在验证迁移后的新文件，或希望覆盖上述自动查找逻辑，请显式传入 `--calib-file`。

### 2. 真机能连上，但动作明显错乱

优先检查：

- 串口接的是不是 follower 臂
- 加载的标定文件是不是当前这台机械臂对应的文件
- 关节装配是否已经发生机械偏移
