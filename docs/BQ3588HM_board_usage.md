# BQ3588 HM (OpenHarmony) 开发板烧录与调试全指南

## 第一阶段：HDC 调试工具准备
HDC (Hardware Device Connector) 是与 OpenHarmony 设备交互的核心工具。在烧录前，建议先在主机上准备好该工具。

1.  **下载全量 SDK：**
    * 访问每日构建 (DailyBuild) 页面：[OpenHarmony DailyBuild](https://dcp.openharmony.cn/workbench/cicd/dailybuild/detail/component)。
    * **项目选择**：`openharmony`；**下载包**：`ohos-sdk-full`。
2.  **获取 HDC 工具：**
    * 解压下载的 SDK 后，从其中的 `toolchains` 目录获取 HDC。
    * Windows 主机通常使用 `windows/toolchains/hdc.exe`。
    * Linux / Ubuntu 主机通常使用 `linux/toolchains/hdc`，或解压后的 `<sdk-root>/toolchains/hdc`。
    * 本仓库当前实验环境验证过的 Linux 路径示例：`/home/xqw/Research/oh_sdk/toolchains/hdc`。
3.  **运行方式 (二选一)：**
    * **方案 A (推荐)：** 配置全局环境变量。把 SDK 的 `toolchains` 目录加入 `PATH`，并写入 shell 启动文件持久化。
      * Bash 示例：`echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.bashrc && source ~/.bashrc`
      * Zsh 示例：`echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.zshrc && source ~/.zshrc`
      * 完成后应能直接执行：`hdc list targets`
    * **方案 B (快捷)：** 无需配置环境。直接在 `toolchains` 文件夹空白处，按住 `Shift` 并右键，选择“在此处打开 PowerShell/终端”即可就地使用。

## 第二阶段：驱动安装与硬件连接
烧录前必须安装瑞芯微底层的 USB 通讯驱动。

1.  **驱动安装：**
    * 下载瑞芯微驱动助手 [DriverAssitant V5.1.1](http://www.mcuzone.com/down/Software.asp?ID=10000617)。
    * 直接运行 `DriverInstall.exe`，点击“驱动安装”。若曾安装过旧版本，建议先点击“驱动卸载”再重新安装。
2.  **物理连线：**
    * 使用数据线（Type-C to Type-C，或 Type-A to Type-C 均可）连接电脑与开发板的 **Type-C (OTG)** 接口。

## 第三阶段：镜像烧录流程 (重点)
烧录过程涉及手动路径匹配，请务必仔细核对。

1.  **镜像获取：**
    * 参考 [OpenHarmony Robot Docs](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md) 中的最新镜像链接下载 BQ3588 专用镜像。
2.  **进入烧录模式：**
    * **软件切换：** 在终端执行 `hdc target boot loader` 强制设备重启进入烧录模式。
    * **硬件强制：** 若系统无法启动，按住板载 **Recovery** 键不放，点击 **Reset** 键复位，2 秒后松开 Recovery 键。工具下方显示“发现一个 LOADER 设备”即成功。
3.  **配置烧录项：**
    * 打开 `RKDevTool`，右键点击列表空白处选择“导入配置”，选择镜像目录下的 `config.cfg`。
    * **关键步骤：** 工具自动填充的路径通常不正确。必须逐一点击各镜像项（如 `uboot`, `system`, `vendor` 等）右侧的路径栏，**手工选择**本地对应的 `.img` 文件。
4.  **执行烧录：**
    * 确认第一行 `Loader` 对应的是 `MiniLoaderAll.bin`。
    * 勾选所有需要更新的项，点击 **“执行”**。

## 第四阶段：高可靠网络调试 (TCP 模式)
为避免大文件传输导致的 USB 僵尸会话，建议切换到局域网 TCP 调试。

1.  **开启监听：** 在 USB 连接状态下执行 `hdc tmode port 8710`。
2.  **获取 IP：** 执行 `hdc shell ifconfig` 查看开发板当前局域网 IP。
3.  **远端连接：** 执行 `hdc tconn <board-ip>:8710`。
4.  **操作习惯：** 建议使用 `hdc -t <board-ip>:8710 shell` 明确指定目标设备，确保自动化交互的稳定性。

## 第五阶段：SSH 登录与公钥配置

在当前实验环境中，开发板已经验证过可通过 SSH 登录；推荐先用 HDC/TCP 完成初始化，再切换到 SSH 做日常操作。

### 1. 连接前提

1. 先确认开发板已经联网，并能通过 HDC/TCP 访问。
2. 在板端至少执行过一次：

   ```sh
   cd /data
   . ./ros2ohos.env
   ```

   当前板子的 `/data/sysdeps.env` 已经补过 `mount -o remount,rw /`，因此这一步会同时把 SSH 依赖目录准备好。
3. 获取开发板当前局域网 IP 后，主机侧可直接测试：

   ```sh
   ssh root@<board-ip>
   ```

### 2. 密码登录说明

- 用户名使用 `root`。
- 是否能直接使用密码登录，取决于当前镜像里的 SSH 配置与 root 密码状态。
- 如果密码未知或镜像未启用密码登录，**不要反复猜密码**，直接改走公钥方式。

### 3. 推荐方式：通过 HDC 上传 SSH 公钥

先在主机上准备公钥（如尚未生成）：

```sh
ssh-keygen -t ed25519 -C "ibrobot-bq3588hm"
```

然后通过 HDC 推送公钥并写入 `authorized_keys`：

```sh
HDC_BIN=<sdk-root>/toolchains/hdc
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send ~/.ssh/id_ed25519.pub /data/local/id_ed25519.pub
"$HDC_BIN" -t "$HDC_TARGET" shell '
mkdir -p /root/.ssh &&
cat /data/local/id_ed25519.pub >> /root/.ssh/authorized_keys &&
chmod 700 /root/.ssh &&
chmod 600 /root/.ssh/authorized_keys
'
```

完成后，从主机直接登录：

```sh
ssh root@<board-ip>
```

### 4. 建议的使用习惯

- **大文件传输 / 自动化脚本**：优先使用 HDC/TCP。
- **日常命令行操作 / 多终端调试**：优先使用 SSH。
- 如果 SSH 临时失效，可先回到 HDC 执行：

  ```sh
  cd /data
  . ./ros2ohos.env
  ```

  然后再重新尝试 `ssh root@<board-ip>`。
