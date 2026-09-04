# SD3403 SDK 与 IB-Robot 部署资源

本文整理 SD3403（Hi3403V100 / SS928V100）相关 SDK、ACL/NNIE 工具链及官方 IB-Robot 部署入口。

## 资源链接

| 内容 | 链接 |
| --- | --- |
| ModelZoo 贡献样例 | [samples/contribute](https://gitcode.com/HiSpark/modelzoo/tree/master/samples/contribute) |
| SD3403 开发环境与 SDK | [Hi3403V100 开发环境搭建](https://gitcode.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100开发环境搭建.md) |
| SDK 安装说明 | [openEuler Embedded SDK](https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/getting_started/index.html#install-openeuler-embedded-sdk) |
| SS928V100 工具链 | [HiSpark/ss928v100_gcc](https://gitcode.com/HiSpark/ss928v100_gcc) |
| ACL/OM 示例 | [ACT/NNN](https://gitcode.com/HiSpark/modelzoo/tree/master/samples/contribute/ACT/NNN) |
| SVP/NNIE 示例 | [ACT/SVP_NNN](https://gitcode.com/HiSpark/modelzoo/tree/master/samples/contribute/ACT/SVP_NNN) |
| 官方 IB-Robot | [openEuler/IB_Robot](https://gitcode.com/openeuler/IB_Robot) |
| IB-Robot 推理服务 | [inference_service](https://gitcode.com/openeuler/IB_Robot/tree/master/src/inference_service) |
| IB-Robot 模型工具 | [model_utils](https://gitcode.com/openeuler/IB_Robot/tree/master/src/model_utils) |

## 名称对应

- `Hi3403V100` / `SS928V100`：ModelZoo 使用的平台名称。
- `sd3403`：IB-Robot 中使用的目标 SoC 名称。
- `hisilicon`：IB-Robot 的推理 backend 名称。

官方 IB-Robot 当前以 [master 分支](https://gitcode.com/openeuler/IB_Robot/tree/master)作为代码入口。实际部署时建议同时记录使用的 commit、SDK 版本和 vendor 编译器版本。

## 工具链说明

- **ACL/OM**：典型流程为 `ONNX -> ATC -> OM -> ACL`，参考 [ACT/NNN README](https://gitcode.com/HiSpark/modelzoo/blob/master/samples/contribute/ACT/NNN/README.md)。
- **SVP/NNIE**：使用海思 SVP/NNIE 运行时接口，参考 [ACT/SVP_NNN README](https://gitcode.com/HiSpark/modelzoo/blob/master/samples/contribute/ACT/SVP_NNN/README.md)。两套运行时 API 不应混用。

## IB-Robot 部署示例

### 1. 导出 ACT ONNX

```bash
source .shrc_local
python3 src/model_utils/model_utils/export_onnx_hisilicon.py \
    --policy_path /path/to/act_bundle \
    --policy_type act \
    --device cpu \
    --bundle_output /path/to/compiled_bundle
```

使用目标平台 vendor 工具链将 ONNX 编译为 OM、worker executable 和 ABI JSON。

### 2. 打包 deployment

```bash
source .shrc_local
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/compiled_bundle \
    --deployment sd3403 \
    --backend hisilicon \
    --target-soc sd3403 \
    --target-runtime hisilicon-worker \
    --spec /path/to/hisilicon-package-spec.json
```

### 3. 启动推理

```bash
source .shrc_local
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:="$WORKSPACE/src/robot_config/config/robots/so101_single_arm.yaml" \
    model_path:="$WORKSPACE/models/act_sd3403_bundle" \
    deployment:=sd3403 \
    pipeline_id:=policy
```

其中 `deployment` 必须与 `inference_manifest.json` 中的命名 deployment 一致。
