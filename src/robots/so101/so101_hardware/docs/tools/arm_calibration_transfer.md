# arm_calibration_transfer

SO-101 旧版标定数据迁移工具。它用于把旧机械臂上的 `homing_offset`、`drive_mode` 等标定信息映射到当前标定文件格式中，生成一个可直接给 IB_Robot 使用的新校准文件。

## 适用场景

- 旧数据、旧模型是按老机械臂的标定参数录制或训练的。
- 新机械臂已经完成当前版本的标准标定，但直接使用旧数据时动作会有系统性偏差。
- 希望尽量复用旧数据或旧策略，而不是重新录制全部样本。

## 核心思路

工具会同时读取两类 JSON 文件：

- 旧标定文件：提供旧机械臂的 `homing_offset` 和 `drive_mode`
- 新标定文件：提供当前机械臂的关节范围模板 `range_min` / `range_max`

随后按关节重新计算新的范围中心，并做两类保护：

- 若旧标定启用了反向驱动，会自动翻转范围方向
- 若计算后范围下界小于 0，会整体平移到合法区间

生成结果仍然是 IB_Robot 当前使用的 follower 标定 JSON 结构，便于直接替换或挂接到 `robot_config` 中。

## 输入与输出

输入目录约定：

- `old_dir_path`：旧版本标定文件目录
- `new_dir_path`：当前版本标定文件目录

文件名约定：

- 旧文件通常形如 `<old_arm_id>.json`
- 新模板文件通常形如 `<new_arm_id>.json`

输出文件：

- 默认输出为 `finetune_<new_arm_id>.json`
- 输出目录仍在 `new_dir_path` 下

## ROS 2 环境准备

本工具本身不依赖运行中的 ROS 图，但推荐在已完成编译的 ROS 2 工作区中通过 `ros2 run` 调用。

如果你在 IB_Robot 仓根目录操作，推荐：

```bash
cd /path/to/IB_Robot
source .shrc_local
```

如果你使用标准 ROS 2 工作区方式，至少需要：

```bash
source /opt/ros/humble/setup.bash
source /path/to/IB_Robot/install/setup.bash
```

如果此前没有编译过 `so101_hardware`，先执行：

```bash
cd /path/to/IB_Robot
source .shrc_local
colcon build --merge-install --symlink-install --packages-select so101_hardware
```

## 使用方式

### 单组新旧机械臂映射

```bash
cd /path/to/IB_Robot
source .shrc_local

ros2 run so101_hardware arm_calibration_transfer \
  --arms_name_list='[["single_follower_arm","legacy_follower_arm"]]' \
  --new_dir_path /tmp/calibration/new \
  --old_dir_path /tmp/calibration/old
```

含义如下：

- `single_follower_arm`：当前仓希望使用的新机械臂 ID
- `legacy_follower_arm`：旧文件中对应的机械臂 ID
- `/tmp/calibration/new/single_follower_arm.json`：当前模板文件
- `/tmp/calibration/old/legacy_follower_arm.json`：旧版本文件

执行完成后会生成：

```text
/tmp/calibration/new/finetune_single_follower_arm.json
```

### 一次迁移多组机械臂

多组机械臂映射可直接放在同一个 JSON 列表里：

```bash
ros2 run so101_hardware arm_calibration_transfer \
  --arms_name_list='[["left_follower","legacy_left"],["right_follower","legacy_right"]]' \
  --new_dir_path /data/new_calibration \
  --old_dir_path /data/old_calibration
```

## 在 IB_Robot 中如何使用迁移结果

迁移工具只负责“生成新的标定文件”，不会自动改写机器人配置。常见接法有两种：

### 方式一：直接替换默认 follower 标定文件

如果你希望现有配置不改路径，可以将结果文件复制或重命名为：

```text
~/.calibrate/so101_follower_calibrate.json
```

这是 `so101_hardware` 和默认 `robot_config` 配置最常见的 follower 标定文件位置。

### 方式二：在 robot_config 中显式指定新文件

例如修改机器人配置中的：

```yaml
robot:
  ros2_control:
    calib_file: /abs/path/to/finetune_single_follower_arm.json
```

对应文件可参考：

- `src/robot_config/config/robots/so101_single_arm.yaml`

## 建议的验证方式

迁移完成后，建议至少做一次关节校验，而不是直接上任务。

硬件环境可进一步运行：

```bash
ros2 run so101_hardware arm_calibration_checker \
  --port /dev/ttyACM0 \
  --calib-file /abs/path/to/finetune_single_follower_arm.json
```

这样可以确认迁移后的标定文件在 30° / 60° / 90° 等检查点是否仍与物理姿态一致。

## 常见问题

### 1. 提示找不到模板文件或旧文件

请优先检查：

- `new_dir_path/<new_arm_id>.json` 是否存在
- `old_dir_path/<old_arm_id>.json` 是否存在

### 2. 提示迁移结果非法

工具会对生成结果做范围合法性校验。如果报错，通常说明：

- 旧标定文件内容不完整
- 新模板文件字段缺失
- 某些关节迁移后超出了 `0..4095` 合法范围

这时应回头检查源文件，而不是强行使用输出结果。

### 3. 迁移完成但机械臂动作仍不对

这通常不是工具“没生效”，而是以下问题之一：

- 运行时实际加载的并不是新生成的标定文件
- 新旧机械臂装配差异过大，单纯做范围重映射仍不足以完全对齐
- 旧数据本身录制时就带有偏差

建议先确认 `robot_config` 的 `calib_file` 指向，再结合 `arm_calibration_checker` 做物理核验。
