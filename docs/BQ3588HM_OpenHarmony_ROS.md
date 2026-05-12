# BQ3588HM OpenHarmony ROS 安装与 IB_Robot 交叉编译说明

本文档面向 **Bearkey BQ3588HM + OpenHarmony 5.1** 的使用场景，整理两件事：

1. 如何在板端安装官方提供的 OpenHarmony ROS 2 Humble 运行时。
2. 如何在 Ubuntu 主机上交叉编译 IB_Robot 的 ROS 包，并部署到开发板。

本文档不重复说明烧录、HDC/SSH 联网等基础内容；这些内容请先参考：

- [BQ3588HM_board_usage.md](./BQ3588HM_board_usage.md)

## References

- 官方二进制使用文档（包含 **开发板镜像**、**ROS 系统依赖二进制包**、**ROS 2 Humble 运行时二进制包**）：
  - <https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md>
- 官方 Docker 交叉编译文档：
  - <https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/docker-build.md>
- 其中与本文最相关的小节：
  - `docker-build.md` -> `用户自定义 ROS2 包/项目编译和使用`

> 说明：下面的下载项、目录约定和交叉编译流程，都是在官方文档基础上，结合本仓库的
> `scripts/openharmony/build_ibrobot_oh_custom.sh` 做的 IB_Robot 化整理。

## 缺少前置条件时先看哪里

- **如果主机侧没有可用的 `hdc`**：
  - 先看 `docs/BQ3588HM_board_usage.md` → **第一阶段：HDC 调试工具准备**
  - 再看本文 **1.4 OpenHarmony ROS SDK**
- **如果主机侧没有预设 `OH_ROOT` / `OH_DOWNLOAD_ROOT` / `OH_CUSTOM_ROOT`**：
  - 先看本文 **第 2 节：统一放到一个外部目录**
  - 再看本文 **第 4 节：`build_ibrobot_oh_custom.sh` 里的变量到底对应什么**

如果这些变量在当前 shell 里不存在，不要让自动化工具猜目录；请先按本文整理目录并导出
`OH_ROOT`，或者在脚本里显式传 `--root`、`--sdk-tar`、`--sysdeps-tar`、`--humble-tar`。

对于 `hdc`，也推荐不要在脚本里写死某个用户私有绝对路径；更稳妥的做法是先把 SDK 的
`toolchains` 目录导出到 `PATH`，并把这条导出写入 `~/.bashrc` 或 `~/.zshrc`，之后直接
使用 `hdc` 命令。

## 1. 需要准备的下载内容

对于 BQ3588HM（`aarch64`）场景，建议至少准备下面四类文件：

| 类别 | 官方用途 | 官方来源 |
| --- | --- | --- |
| BQ3588HM OpenHarmony 镜像 | 烧录开发板 | `usage.md` 中的 `oh5.1 bq3588 build` |
| `ohos-*-sysdeps-*.tar.gz` | OpenHarmony ROS 系统依赖包 | `usage.md` 中的 `ohos-sysdeps-xxx.tar.gz` |
| `ohos-humble-build-*.tar.gz` | OpenHarmony ROS 2 Humble 运行时发行版 | `usage.md` 中的 `ohos-humble-build-xxx.tar.gz` |
| `ohos-sdk-18-linux-aarch64-*.tar.gz` | 主机交叉编译时使用的 OHOS SDK | `docker-build.md` 中的 `ohos-ros-sdk-build` |

另外还需要一个 Docker 编译镜像：

```bash
docker pull voxelsky/ohos-ros-humble-builder:v0.1.5
```

官方文档当前给出的下载入口如下：

### 1.1 BQ3588HM 镜像

- 百度网盘：<https://pan.baidu.com/s/1BA5F8Ph7gpsrawpzvPEofA?pwd=kaq4>（提取码：`kaq4`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/52224e51bcb98be6ab043c5846ddfb7f>（提取码：`m7fp`）

### 1.2 OpenHarmony ROS 系统依赖包

- 百度网盘：<https://pan.baidu.com/s/14b4YyQWxIBdKj2ZOu2I-VQ?pwd=sb3y>（提取码：`sb3y`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/0bfcc408563cba4940905ba54607da38>（提取码：`dch8`）

### 1.3 OpenHarmony ROS 2 Humble 发行版

- 百度网盘：<https://pan.baidu.com/s/1562-HKLWZXbkNeMVHa3PNg?pwd=5tuy>（提取码：`5tuy`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/9fe41dd3ac1fc712b9157a46778545c3>（提取码：`w5kz`）

### 1.4 OpenHarmony ROS SDK

- 百度网盘：<https://pan.baidu.com/s/168iE3OZT-5qswn24tf1oAA?pwd=k8wk>（提取码：`k8wk`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/7c24241fbfb00c683ecedf73d6366fa4>（提取码：`e1hd`）

> 建议总是选择**日期最新且架构为 `aarch64`** 的文件。不要混用 `x86_64` 和 `aarch64`
> 产物。

## 2. 统一放到一个外部目录：推荐的主机目录布局

更推荐的做法是：**由用户自己指定一个统一的 OpenHarmony 主机目录**，把下载内容和交叉编译
目录都放在这里，并通过脚本参数 `--oh-root` 传给
`scripts/openharmony/build_ibrobot_oh_custom.sh`，而不是散落在多个临时目录中。

例如把这个统一目录记为 `<OH_ROOT>`：

```text
<OH_ROOT>/
├── downloads/
│   ├── images/
│   │   └── oh5.1-bq3588-build-...
│   ├── sdk/
│   │   └── ohos-sdk-18-linux-aarch64-....tar.gz
│   ├── sysdeps/
│   │   └── ohos-18-sysdeps-aarch64-....tar.gz
│   └── runtime/
│       └── ohos-humble-build-aarch64-....tar.gz
└── custom_build_root/
    ├── install/
    ├── ibrobot_oh_ws/
    │   └── src/
    ├── ohos-robot-toolchain/
    │   └── 18/native/
    ├── ros_ros2_base/
    └── version/
```

下文统一把这个目录记为：

```bash
export OH_ROOT="<your-unified-oh-root>"
export OH_DOWNLOAD_ROOT="$OH_ROOT/downloads"
```

脚本会默认按下面这个约定从 `OH_ROOT` 派生路径：

```text
OH_DOWNLOAD_ROOT = $OH_ROOT/downloads
OH_CUSTOM_ROOT   = $OH_ROOT/custom_build_root
```

这里的 `OH_ROOT` / `OH_DOWNLOAD_ROOT` / `OH_CUSTOM_ROOT` 都是**主机侧交叉编译变量**，
不是开发板上的环境变量。如果当前 shell 里没有这些变量，这通常是正常的；请先按本节准备
目录布局并自行导出，或者在调用脚本时改用显式参数，不要依赖“自动猜测主机目录”。

## 3. 板端安装 OpenHarmony ROS 2 Humble

这一部分对应官方 `usage.md` 的主流程。

### 3.1 将运行时包上传到开发板

如果板端 `/data` 里还没有这两个包，请先上传：

- `ohos-humble-build-*.tar.gz`
- `ohos-*-sysdeps-*.tar.gz`

例如使用 HDC：

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_DOWNLOAD_ROOT/runtime/ohos-humble-build-aarch64-....tar.gz" \
  /data/ohos-humble-build-aarch64.tar.gz

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_DOWNLOAD_ROOT/sysdeps/ohos-18-sysdeps-aarch64-....tar.gz" \
  /data/ohos-18-sysdeps-aarch64.tar.gz
```

当前实验环境中，板子上已经验证存在的典型路径是：

- `/data/ohos-humble-build-aarch64-20260115100449.tar.gz`
- `/data/ohos-18-sysdeps-aarch64-20260115.tar.gz`

### 3.2 在板端解压并加载 ROS 环境

在开发板上执行：

```sh
cd /data
tar -zxpvf ohos-humble-build-aarch64.tar.gz
tar -zxpvf ohos-18-sysdeps-aarch64.tar.gz

# 注意：必须在 ros2ohos.env 所在目录执行
. ./ros2ohos.env
```

然后检查：

```sh
ros2 topic list
python3 --version
python3 -m pip --version
```

### 3.3 BQ3588HM 板子的额外注意事项

对于本仓库当前验证过的 BQ3588HM 开发板：

- `ros2ohos.env` 位于 `/data/ros2ohos.env`
- ROS 运行时目录通常是 `/data/install`
- sysdeps 目录通常是 `/data/out`
- 当前板端的 `/data/sysdeps.env` 已补过 `mount -o remount,rw /`，因此执行
  `. ./ros2ohos.env` 时可以正常准备 SSH 相关目录

如果你需要 HDC TCP 地址、SSH、公钥登录、只读根文件系统等细节，请继续看：

- [BQ3588HM_board_usage.md](./BQ3588HM_board_usage.md)

## 4. `build_ibrobot_oh_custom.sh` 里的变量到底对应什么

`scripts/openharmony/build_ibrobot_oh_custom.sh` 是我们把官方
`docker-build.md -> 用户自定义 ROS2 包/项目编译和使用` 流程，封装成 IB_Robot 专用脚本后的实现。

推荐把 `OH_ROOT` 作为这个脚本的一级输入，再由脚本自动展开出下载目录和交叉编译目录。

它开头定义的变量，建议理解为下面这张表：

| 变量 | 含义 | 推荐值 |
| --- | --- | --- |
| `OH_ROOT` | OpenHarmony 主机统一根目录 | `<your-unified-oh-root>` |
| `OH_DOWNLOAD_ROOT` | 下载内容总目录 | `$OH_ROOT/downloads` |
| `OH_CUSTOM_ROOT` | 交叉编译总根目录 | `$OH_ROOT/custom_build_root` |
| `OH_CUSTOM_WS` | 放 IB_Robot ROS 工作区的目录 | `$OH_CUSTOM_ROOT/ibrobot_oh_ws` |
| `OH_CUSTOM_SRC` | 交叉编译工作区的 `src/` | `$OH_CUSTOM_WS/src` |
| `OH_CUSTOM_TOOLCHAIN_ROOT` | 解压 OHOS SDK 的目录 | `$OH_CUSTOM_ROOT/ohos-robot-toolchain` |
| `OH_CUSTOM_SDK_TAR_GLOB` | 官方 `ohos-sdk-18-linux-aarch64-*.tar.gz` 的位置 | `$OH_DOWNLOAD_ROOT/sdk/...tar.gz` |
| `OH_CUSTOM_SYSDEPS_TAR_GLOB` | 官方 `ohos-*-sysdeps-*.tar.gz` 的位置 | `$OH_DOWNLOAD_ROOT/sysdeps/...tar.gz` |
| `OH_CUSTOM_HUMBLE_TAR_GLOB` | 官方 `ohos-humble-build-*.tar.gz` 的位置 | `$OH_DOWNLOAD_ROOT/runtime/...tar.gz` |
| `OH_CUSTOM_ROS2_BASE_REPO` | 自动克隆的 `ros_ros2_base` 仓库目录 | `$OH_CUSTOM_ROOT/ros_ros2_base` |
| `OH_CUSTOM_VERSION_REPO` | 自动克隆的 `version` 仓库目录 | `$OH_CUSTOM_ROOT/version` |
| `OH_CUSTOM_PREFIX` | 板端最终安装前缀 | `/data/ibrobot/install` |
| `OH_CUSTOM_IMAGE` | Docker 构建镜像 | `voxelsky/ohos-ros-humble-builder:v0.1.5` |

### 4.1 这些下载内容和变量的对应关系

你下载的文件，建议这样放：

```text
$OH_DOWNLOAD_ROOT/sdk/ohos-sdk-18-linux-aarch64-....tar.gz
$OH_DOWNLOAD_ROOT/sysdeps/ohos-18-sysdeps-aarch64-....tar.gz
$OH_DOWNLOAD_ROOT/runtime/ohos-humble-build-aarch64-....tar.gz
```

脚本运行时会自动做这些事：

1. 把 `ohos-humble-build-*.tar.gz` 解压到 `OH_CUSTOM_ROOT`，得到 `install/`
2. 把 `ohos-sdk-18-linux-aarch64-*.tar.gz` 解压到 `OH_CUSTOM_TOOLCHAIN_ROOT/18/native`
3. 把 `ohos-*-sysdeps-*.tar.gz` 里的 Python 3.12 / sframe 相关内容 overlay 到 SDK sysroot
4. 自动克隆：
   - `https://gitcode.com/openharmony-robot/ros_ros2_base.git`
   - `https://gitcode.com/openharmony-robot/version.git`
5. 从 IB_Robot 仓库复制交叉编译需要的包到 `OH_CUSTOM_WS/src`

所以这不是“又下载一堆和脚本无关的文件”；相反，这几个 tarball 就是脚本真正需要消费的输入。

## 5. 用我们的脚本交叉编译 IB_Robot

### 5.1 前提条件

主机侧准备好：

- Ubuntu 主机
- Docker
- `voxelsky/ohos-ros-humble-builder:v0.1.5`
- 下载好的：
  - `ohos-sdk-18-linux-aarch64-*.tar.gz`
  - `ohos-*-sysdeps-*.tar.gz`
  - `ohos-humble-build-aarch64-*.tar.gz`

### 5.2 推荐执行方式

先在主机上准备统一根目录：

```bash
export OH_ROOT="<your-unified-oh-root>"
```

然后在 IB_Robot 仓库根目录执行：

```bash
./scripts/openharmony/build_ibrobot_oh_custom.sh \
  --oh-root "$OH_ROOT" \
  --image voxelsky/ohos-ros-humble-builder:v0.1.5 \
  --packages ibrobot_msgs,tensormsg,robot_config,inference_service
```

只要 `OH_ROOT` 下的目录布局符合第 2 节约定，脚本会自动从：

- `$OH_DOWNLOAD_ROOT/sdk/`
- `$OH_DOWNLOAD_ROOT/sysdeps/`
- `$OH_DOWNLOAD_ROOT/runtime/`

解析 SDK、sysdeps 和 Humble runtime tarball。

如果你的文件不在默认布局里，再额外使用：

- `--sdk-tar`
- `--sysdeps-tar`
- `--humble-tar`

做精确覆盖即可。

### 5.3 这条命令实际做了什么

它本质上是在封装官方文档里的这类调用：

```bash
build-ros-humble --custom \
  --wd <项目工作目录> \
  --custom-prefix /data/ibrobot/install \
  --colcon-args --packages-select ibrobot_msgs tensormsg robot_config inference_service
```

脚本会在容器里把下面两个环境变量设好：

```bash
WS_ROOT=/mnt/ohos/tmp
OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
```

并把你的 `OH_CUSTOM_ROOT` 挂载进容器。因此编译产物会直接落回主机目录。

## 6. 编译完成后产物在哪

编译完成后，重点看：

```text
$OH_CUSTOM_ROOT/ibrobot_oh_ws/install
```

这就是需要部署到板端的自定义 ROS 工作区安装结果。

如果你沿用默认前缀 `/data/ibrobot/install`，那推荐的打包和部署方式是：

```bash
cd "$OH_CUSTOM_ROOT/ibrobot_oh_ws"
tar -zcpf ibrobot-oh-install.tar.gz install
```

然后上传到板端：

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_CUSTOM_ROOT/ibrobot_oh_ws/ibrobot-oh-install.tar.gz" \
  /data/ibrobot-oh-install.tar.gz
```

在板端解压：

```sh
cd /data
mkdir -p /data/ibrobot
tar -zxpf ibrobot-oh-install.tar.gz -C /data/ibrobot
ls -lah /data/ibrobot/install
```

最终目录应当是：

```text
/data/ibrobot/install
```

## 7. 板端如何使用交叉编译出来的 IB_Robot 包

先加载官方 ROS for OpenHarmony 运行时：

```sh
cd /data
. ./ros2ohos.env
```

然后加载你自己的工作区：

```sh
cd /data/ibrobot
. install/setup.sh
```

之后就可以执行你部署进去的 ROS 2 包，例如：

```sh
ros2 pkg list | grep -E 'ibrobot_msgs|tensormsg|robot_config|inference_service'
```

如果你还需要在板端跑 `LeRobot + torch` 的 Cloud 推理链路，请继续看：

- [OpenHarmony_thirdparty_pytorch_validation.md](./OpenHarmony_thirdparty_pytorch_validation.md)

## 8. 和官方文档的关系

如果你只关心“下载二进制后怎么在板上用”，官方 `usage.md` 已经足够。

如果你只关心“如何手工用 Docker 交叉编译任意 ROS 项目”，官方 `docker-build.md` 的
`用户自定义 ROS2 包/项目编译和使用` 已经给出了通用流程。

而本文档额外补上的内容是：

1. **把 BQ3588HM 板端 ROS 安装和 IB_Robot 交叉编译放到一条连续流程里。**
2. **把 `build_ibrobot_oh_custom.sh` 的变量和实际下载内容一一对应起来。**
3. **明确建议把 SDK / sysdeps / runtime 放到仓库外的独立目录，而不是 IB_Robot 的 `tmp/`。**
4. **说明编译结果如何落到 `/data/ibrobot/install`，以及板端如何叠加加载。**
