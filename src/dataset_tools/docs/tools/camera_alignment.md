# camera_alignment

基于 ArUco 标记的相机对齐工具。它通过检测当前画面中的多个 ArUco 角点，计算相对参考帧的平均像素误差，并提供实时颜色反馈和“虚影对齐”界面，帮助你把摄像头恢复到训练或基准采集时的位置。

## 适用场景

- 推理前需要确认相机视角没有漂移
- 摄像头支架刚拆装过，担心视角偏了
- 需要快速对齐本机 USB 摄像头或视频设备视角

## 工具能力

当前工具保持原始迁移工具的输入方式，并支持显式指定采集格式：

- `--cameras_index_or_path`：直接从本机摄像头或视频设备读取
- `--width` / `--height`：请求 OpenCV 以指定分辨率打开设备
- `--fps`：请求指定帧率
- `--format`：请求指定采集格式，例如 `MJPG`、`YUYV`

工具支持如下交互：

- `s`：保存当前画面中的 ArUco 角点作为参考基准
- `v`：进入虚影对齐模式
- `q`：退出

默认会生成两个文件：

- `camera_reference_multi.json`：保存参考角点和参考帧尺寸
- `reference_img.png`：保存参考图像

## 运行前提

推荐在 IB_Robot 仓根目录执行：

```bash
cd /path/to/IB_Robot
source .shrc_local
```

如果 `dataset_tools` 尚未编译，先执行：

```bash
cd /path/to/IB_Robot
source .shrc_local
colcon build --merge-install --symlink-install --packages-select dataset_tools
```

工具依赖：

- `python3-opencv`
- OpenCV ArUco 模块 `cv2.aruco`
- `python3-numpy`

如果使用窗口界面，还需要有可用显示环境，例如：

- 物理桌面环境
- 开发容器中的 X11 转发
- VNC / 远程桌面

## ROS 2 中的使用方式

在 ROS 2 环境中直接读取本机视频设备：

```bash
cd /path/to/IB_Robot
source .shrc_local

ros2 run dataset_tools camera_alignment \
  --cameras_index_or_path /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 60 \
  --format MJPG \
  --reference-path /tmp/camera_reference_multi.json \
  --reference-image-path /tmp/reference_img.png
```

如果设备号是整数，也可以写成：

```bash
ros2 run dataset_tools camera_alignment --cameras_index_or_path 0
```

## 使用步骤

### 1. 保存参考基准

把摄像头调到你认可的“黄金位置”，确保画面里能看到 ArUco 码，然后按：

```text
s
```

### 2. 观察误差状态

主界面会持续显示误差状态：

- 绿色：误差小于 3 像素
- 红色：误差大于等于 3 像素
- 黄色：还没有保存参考基准，或当前丢失了 marker

### 3. 进入虚影模式

按：

```text
v
```

退出虚影模式按：

```text
q
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `--cameras_index_or_path` | 本机摄像头索引或设备路径，如 `0`、`/dev/video0` |
| `--reference-path` | 参考角点 JSON 输出路径 |
| `--reference-image-path` | 参考图输出路径 |
| `--width` | 请求采集宽度，单位为像素 |
| `--height` | 请求采集高度，单位为像素 |
| `--fps` | 请求采集帧率 |
| `--format` | 请求四字符采集格式，例如 `MJPG`、`YUYV` |

启动后，工具会在首次读取到画面时打印实际生效的采集参数。如果设备或 OpenCV
后端没有接受请求值，会打印 warning，例如实际分辨率、帧率或 FOURCC 与请求值不一致。

## 参考文件格式

新保存的参考 JSON 会记录参考帧尺寸：

```json
{
  "image_width": 640,
  "image_height": 480,
  "markers": {
    "1": [[...], [...], [...], [...]]
  }
}
```

旧版只包含 marker 映射的 JSON 仍可读取。读取到新版参考文件时，如果当前帧尺寸
与参考帧尺寸不同，工具会提示像素误差处于不同坐标系中，结果不可靠，建议重新保存参考。

## 常见问题

### 1. 提示 OpenCV 没有 aruco

说明当前 OpenCV 构建不包含 `cv2.aruco`。需要安装带 ArUco 模块的 OpenCV 版本。

### 2. 看得到图像，但一直检测不到 marker

请检查：

- 使用的是否为 `DICT_4X4_50` 字典族的 ArUco 码
- marker 是否过小、反光、模糊或被遮挡
- 画面中是否只露出了一部分 marker

### 3. 保存了参考，但下次运行找不到参考文件

建议在调用时显式传入：

- `--reference-path`
- `--reference-image-path`

### 4. 提示参考尺寸和当前画面尺寸不一致

说明当前帧和参考帧不在同一像素坐标系中。请使用相同的 `--width`、`--height`、
`--fps`、`--format` 参数重新打开设备，并按 `s` 重新保存参考基准。
