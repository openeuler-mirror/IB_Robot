# FullSubNet 在 Ascend 310B1 NPU 上的 .om 转换与部署

本文记录 FullSubNet cumulative 218epochs checkpoint 转换为 Ascend 310B1 .om 的全过程：从 fused LSTM 直接导出开始，经历 SB 时延超标与 FB 长序列精度漂移两个问题，最终以「静态展开 LSTM + origin 精度模式」收尾，并集成进 `models/fullsubnet` bundle 的 `ascend_310b` deployment。

> **关键结论：unrolled + origin 组合下，FB 2.78ms + SB 18.48ms ≈ 21.3ms（预算 32ms/block），16 帧门控 cos ≈ 1.0，FB 256 帧漂移有界（cos_min 0.99996），SB 64 帧漂移数值上可忽略（cos ≈ 1.0）。**

## 一、背景

- FullSubNet 是 speech_direction 流水线的语音增强模型，310P 上已有 FB/SB 拆分 stateful OM（`ascend_310p` deployment，cann-8.1.RC1）。
- 310B1（openEuler Embedded aarch64 板，CANN 8.3.RC1）需要自己的 OM 对；两侧共用同一 cumulative checkpoint。
- 转换工作目录：`models/_work/fullsubnet/onnx2om_310b/`（ONNX、GT、板端脚本、报告均保留在此，bundle 只放最终 OM）。

## 二、模型结构与固定 ABI

生产链路把 FullSubNet 拆成两个 stateful 子图，Host 负责 STFT / 归一化 / OLA，OM 只做网络推理；LSTM hidden/cell 作为图的输入输出回环（生产用 ACL dataset bank 双 buffer ping-pong，验证脚本用 host 回放，数值等价）：

| 子图 | 输入 | 输出 | 说明 |
|------|------|------|------|
| FB | `frame[4,2,257]` + `hidden[2,4,512]` + `cell[2,4,512]` | `output[4,2,257]` + `hidden_out` + `cell_out` | LSTM(257→512, 2层) + fc(512→257) + ReLU |
| SB | `frame[1028,2,32]` + `hidden[2,1028,384]` + `cell[2,1028,384]` | `output[1028,2,2]` + `hidden_out` + `cell_out` | LSTM(32→384, 2层) + fc(384→2) |

Host ABI 全部 float32（ATC fp16/origin 模式下 OM IO dtype 仍为 float32，内部计算精度由 `--precision_mode_v2` 决定）。T=2、batch=4（FB）/ 4×257=1028（SB）。

## 三、环境

### 3.1 转换机（x86_64 Ubuntu）

- CANN toolkit 8.3.RC1（`/usr/local/Ascend/ascend-toolkit/8.3.RC1`）
- ATC 需要 stub 库路径（无本地 NPU）：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/runtime/lib64/stub/linux/x86_64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
```

- Python 侧用仓库 venv（`source .shrc_local`），依赖 `fullsubnet` wheel（`from fullsubnet.model import Model`）、`onnx`、`onnxruntime`。
- 静态 shape 导出时 ATC 提示 `This model is irrelevant to the host platform`，即无需 `--host_env_cpu=aarch64`（与 RAM/RAM++ 文档结论一致，只有 dynamic_axes 才嵌入 host 平台信息）。

### 3.2 板端（openEuler Embedded aarch64, Ascend 310B1）

- CANN 8.3.RC1 + driver，NPU 为 `/dev/davinci0`
- pyACL 位于 `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`，配合 `/IB_Robot/venv/bin/python3`（3.11）
- 运行验证脚本需要显式挂 CANN 与 driver 库路径（见第八节命令）

### 3.3 pyACL 8.3 API 差异（踩坑记录）

板端 CANN 8.3 的 pyACL 与常见旧示例代码不同，逐一探测后确认：

| 常见写法（旧） | 本板实际 API |
|---|---|
| `acl.rt.create_context()` | `ctx, ret = acl.rt.create_context(0)`（必须传 device id，返回二元组） |
| `acl.mdl.create_data_buffer(ptr, size)` | `buf, ret = acl.create_data_buffer(ptr, size)`（在顶层 `acl` 模块） |
| `acl.mdl.dataset_add_dataset(ds, buf)` | `acl.mdl.add_dataset_buffer(ds, buf)`（返回 `(dataset, ret)`） |
| `acl.mdl.get_input_size(desc, i)` | `acl.mdl.get_input_size_by_index(desc, i)`（output 同理） |
| `acl.rt.destroy_context()` | `acl.rt.destroy_context(ctx)`（必须传 ctx） |
| — | `acl.mdl.get_desc(desc, model_id)` 返回裸 ret；`get_num_inputs/outputs` 返回裸 int；`get_input_dims` 返回 `(dims_dict, ret)`，shape 在 `dims['dims']` |
| — | `acl.util.numpy_to_ptr` 已弃用，推荐 `acl.util.bytes_to_ptr` |

## 四、基准 GT 与第一版（fused LSTM）导出

### 4.1 冻结 Torch fp32 GT

`export_stateful_onnx.py`：用生产同款 wrapper（`StatefulFullBandT2` / `StatefulSubBandT2`，来自 `fullsubnet_stateful_torch.py`）加载 checkpoint，固定种子（20260907）生成 16 帧随机序列（`randn*0.5`），从零态回放并保存逐帧 `frames / outputs / hiddens / cells`：

- `gt_fb.npz`（16 帧）、`gt_sb.npz`（16 帧）：门控用
- `gt_fb_long.npz`（256 帧，含逐帧状态）、`gt_sb_long.npz`（64 帧，输出 + 末态）：漂移分析用（`gen_long_gt.py`）

所有 ONNX / OM 一律与这套 GT 对比，禁止跨目录拼装参照物。

### 4.2 fused ONNX 导出与门控

`torch.onnx.export` 直接导出 wrapper（LSTM 保持融合算子），输入名 `frame/hidden/cell`，输出名 `output/hidden_out/cell_out`。Torch vs ONNXRuntime（CPU）逐帧回放对比：FB/SB 输出、hidden、cell 余弦全部 1.000000，max_abs ≤ 4.6e-5（FB）/ 6e-6（SB）。**ONNX gate PASS**。

### 4.3 fused ATC 编译与板端首测

`--precision_mode_v2=fp16` 编译两个 OM 成功（FB 7.88MB / SB 3.84MB）。板端回放结果：

| 指标 | FB | SB |
|---|---|---|
| 16 帧 out cos_min | 0.99527 | 0.9999994 |
| execute 时延（100 次均值） | 1.61ms | **40.22ms** |

**两个问题暴露**：SB 40ms 严重超 32ms/block 预算；FB 余弦随帧数下滑（16 帧已到 0.995），提示递归状态漂移。

## 五、问题一：SB fused LSTM 时延 40ms

排查过程：

1. `--op_select_implmode=high_performance` 重编 SB：40.32ms，无效。
2. `batch_first=True` 语义确认无误（SB 为 T=2 / batch=1028，不存在 T=1028 的误解）。
3. 根因判定：**fused ONNX LSTM 算子在 310B1 上映射到低效 kernel**（同样形状的 MatMul+Sigmoid+Tanh 组合远快于 LSTM 算子）。

### 解法：静态展开 LSTM（unrolled）

`export_unrolled_onnx.py` 把 2 步 × 2 层 LSTM 手工展开成 `addmm` 门控 + `sigmoid/tanh` + 逐元素更新的静态图（权重作为 buffer 固化）。等价性验证：

- fp64 下与 `nn.LSTM` 逐元素最大差 **6.9e-15**（数学等价）
- fp32 下 ≤ 3.4e-5（累加顺序噪声，量级正常）
- unrolled ONNX vs GT：cos = 1.000000，**ONNX gate PASS**

板端结果（fp16）：FB 0.95ms、SB **10.27ms**（fused 的 1/4），FB+SB ≈ 11.2ms 进入预算。

## 六、问题二：FB fp16 长序列漂移

用 256 帧（FB）/ 64 帧（SB）长序列回放考察漂移是否有界：

| 变体 | FB 256 帧 out cos_min | FB state cos_min | 结论 |
|---|---|---|---|
| unrolled fp16 | 0.7995 | 0.9453 | 漂移无界，不可用 |
| unrolled cube_fp16in_fp32out | 0.2146 | 0.9546 | 更差 |
| unrolled + hi/lo split matmul（cfio） | 0.2573 | 0.9559 | 补偿无效 |

排查链路（关键推理）：

1. cfio 与 fp16 漂移幅度接近 → 瓶颈不在 cube 输入 cast。
2. hi/lo split（`a@a_hi + a_lo@a` 的精确补偿分解，恢复 cube 输入 ~22 bit 尾数）在 cfio 下依旧漂移 → **剩余误差来自 vector 算子：LSTM 状态更新 `c = f*c + i*g`、`h = o*tanh(c)` 在 fp16/cfio 模式下都以 fp16 执行，每步 ~2^-11 的相对误差在递归中不断注入并累积**。
3. `origin` 精度模式让 vector 算子保持 fp32；此前 fused 图上 origin 编译失败（LSTM 算子内部含 cube MatMul，310B1 cube 不支持 fp32），而 **unrolled 图没有 LSTM 融合算子，origin 可编译成功**。

### 最终组合：unrolled + origin

| 指标 | 数值 |
|---|---|
| FB 16 帧门控 out cos_min | 0.99999999989（max_abs 8.1e-5） |
| FB 256 帧漂移 out cos_min | **0.99996**（有界；state cos_min 0.99985） |
| SB 16 帧门控 out cos_min | 0.99999999999956（max_abs 7.0e-6） |
| SB 64 帧漂移 out cos_min | ≈ 1.0（末态 hidden cos 0.99999999999984） |
| FB / SB execute p50 | 2.76ms / 18.48ms，合计 ≈ 21.3ms |

hi/lo split 版本（+0.00002 余弦收益、多一倍 matmul）与 origin 搭配收益边际，被放弃；最终 pair 即朴素 unrolled 图 + `--precision_mode_v2=origin`。

## 七、候选对比总表

| 候选 | 精度模式 | FB 长序列 cos_min | SB 长序列 cos_min | FB+SB 时延 | 结论 |
|---|---|---|---|---|---|
| fused LSTM | fp16 | （16 帧已 0.995，趋势恶化） | 0.99998（64f） | 1.6+40.2=41.8ms | ❌ SB 超预算 |
| fused LSTM | origin | 编译失败（cube fp32） | - | - | ❌ |
| unrolled | fp16 | 0.80 | 0.99998 | 0.95+10.3=11.2ms | ❌ FB 漂移 |
| unrolled | cube_fp16in_fp32out | 0.21 | ≈1.0 | 1.2+16.9=18.1ms | ❌ FB 漂移更差 |
| unrolled + split | cube_fp16in_fp32out | 0.26 | - | - | ❌ 补偿无效 |
| unrolled + split | origin | 0.99998 | - | 3.5+?ms | 收益边际，弃 |
| **unrolled** | **origin** | **0.99996** | **≈1.0** | **2.8+18.5=21.3ms** | ✅ 采用 |

## 八、板端验证

### 8.1 门控 + 时延（`board_om_verify.py`）

```bash
# 板端（192.168.136.127）
cd /root/fullsubnet_310b_verify
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages \
  /IB_Robot/venv/bin/python3 board_om_verify.py \
  --fb-om fullsubnet_cum_stateful_fb_b4_t2_310b_unrolled_origin.om \
  --sb-om fullsubnet_cum_stateful_sb_b4_t2_310b_unrolled_origin.om
```

- OM ABI 实测与 manifest 一致：3 in / 3 out，`frame/hidden/cell`，float32，字节数精确（FB 8224/16384/16384；SB 263168/3158016/3158016）
- 16 帧逐帧回放（状态回环）vs GT：FB/SB 输出、hidden、cell cos ≈ 1.0，**GATE PASS**
- 时延（100 次 execute）：FB mean 2.78 / p95 2.91ms；SB mean 18.48 / p95 18.55ms

### 8.2 长序列漂移（`board_drift.py`）

```bash
... board_drift.py --om <fb.om> --gt gt_fb_long.npz --frame-shape 4,2,257 --state-shape 2,4,512 --output-shape 4,2,257
... board_drift.py --om <sb.om> --gt gt_sb_long.npz --frame-shape 1028,2,32 --state-shape 2,1028,384 --output-shape 1028,2,2
```

结果 JSON 已归档：`models/_work/fullsubnet/onnx2om_310b/{board_verify_result.json, fb_drift_origin_256.json, sb_drift_origin_64.json}`。

## 九、Bundle 集成

最终 OM 复制入 bundle（改名去掉 `unrolled`，ABI 与实现细节无关）：

```text
models/fullsubnet/artifacts/ascend_310b/fullsubnet/
├── fullsubnet_cum_stateful_fb_b4_t2_310b_origin.om   (15.4MB, sha256 3e1a0af0…42a1)
└── fullsubnet_cum_stateful_sb_b4_t2_310b_origin.om   (7.5MB,  sha256 c1b920dd…c061)
```

代码与配置同步（git 跟踪部分）：

- `src/voice_asr_service/voice_asr_service/package_fullsubnet_bundle.py`：新增 `_ascend_310b_deployment`（`soc=Ascend310B1`，`runtime_abi=cann-8.3.RC1`），OM 路径纳入 required 资产
- `models/fullsubnet/inference_manifest.json`：新增 `ascend_310b` deployment（revision 1；310p/torch 部署不变）
- `scripts/verify_speech_direction_assets.py`：fullsubnet 的 deployment 清单加入 `ascend_310b`
- 测试：`test_audio_contract_validation.py`（fake bundle 补 310B 资产）、`test_model_sessions.py`（参数化覆盖 `ascend_310b` + `cann-8.3.RC1`）
- `src/voice_asr_service/README.md`：资产下载与 deployment 说明

验证：

```bash
source .shrc_local
python3 scripts/verify_speech_direction_assets.py --bundle fullsubnet   # 全 deployment 结构 + SHA-256 OK
python3 -m pytest src/voice_asr_service/test/test_audio_contract_validation.py \
                  src/voice_asr_service/test/speech_direction/test_model_sessions.py  # 16 passed
```

launch 侧无需改动：`ascend_310b` 是 bundle 内 deployment，通过 `custom` profile + `fullsubnet_deployment:=ascend_310b` 显式选择（默认 `ascend_310p` profile 不变；silero-vad 尚无 310B deployment，故未加整机 `ascend_310b` profile）。

## 十、复现命令（转换机）

```bash
source .shrc_local
# 1. fused 导出 + GT（历史路径，用于对比）
python3 models/_work/fullsubnet/onnx2om_310b/export_stateful_onnx.py
python3 models/_work/fullsubnet/onnx2om_310b/compare_onnx_gt.py
# 2. unrolled 导出（最终图源）
python3 models/_work/fullsubnet/onnx2om_310b/export_unrolled_onnx.py
# 3. 长序列 GT
python3 models/_work/fullsubnet/onnx2om_310b/gen_long_gt.py

# 4. ATC（310B1, origin）
source /usr/local/Ascend/ascend-toolkit/latest/bin/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/runtime/lib64/stub/linux/x86_64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
cd models/_work/fullsubnet/onnx2om_310b
atc --framework=5 --model=fullsubnet_cum_stateful_fb_b4_t2_unrolled.onnx \
    --output=fullsubnet_cum_stateful_fb_b4_t2_310b_unrolled_origin \
    --soc_version=Ascend310B1 --precision_mode_v2=origin \
    --input_shape="frame:4,2,257;hidden:2,4,512;cell:2,4,512"
atc --framework=5 --model=fullsubnet_cum_stateful_sb_b4_t2_unrolled.onnx \
    --output=fullsubnet_cum_stateful_sb_b4_t2_310b_unrolled_origin \
    --soc_version=Ascend310B1 --precision_mode_v2=origin \
    --input_shape="frame:1028,2,32;hidden:2,1028,384;cell:2,1028,384"

# 5. bundle 集成（先切回 workspace 根目录；两条显式复制，禁止 brace expansion 误用）
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
cp models/_work/fullsubnet/onnx2om_310b/fullsubnet_cum_stateful_fb_b4_t2_310b_unrolled_origin.om \
   models/fullsubnet/artifacts/ascend_310b/fullsubnet/fullsubnet_cum_stateful_fb_b4_t2_310b_origin.om
cp models/_work/fullsubnet/onnx2om_310b/fullsubnet_cum_stateful_sb_b4_t2_310b_unrolled_origin.om \
   models/fullsubnet/artifacts/ascend_310b/fullsubnet/fullsubnet_cum_stateful_sb_b4_t2_310b_origin.om
source .shrc_local && python3 -m voice_asr_service.package_fullsubnet_bundle --bundle-root models/fullsubnet
```

## 十一、文件清单

`models/_work/fullsubnet/onnx2om_310b/`（保留物）：

| 文件 | 说明 |
|---|---|
| `export_stateful_onnx.py` / `export_unrolled_onnx.py` / `gen_long_gt.py` | 导出与 GT 冻结脚本 |
| `compare_onnx_gt.py` | Torch vs ONNX 门控 |
| `board_om_verify.py` / `board_drift.py` | 板端门控/时延/漂移脚本 |
| `fullsubnet_cum_stateful_{fb,sb}_b4_t2_unrolled.onnx` | 最终图源 |
| `fullsubnet_cum_stateful_{fb,sb}_b4_t2_310b_unrolled_origin.om` | 最终 OM（bundle 内同内容） |
| `board_verify_result.json` / `fb_drift_origin_256.json` / `sb_drift_origin_64.json` | 板端结果 |
| `reports/decisions.jsonl` / `reports/summary.md` | 决策账本与汇总 |

GT npz 与被拒绝的候选 OM/ONNX 已清理。

## 十二、经验总结

1. **310B1 fused LSTM 慢 → unroll**：fused ONNX LSTM 在 310B1 上 SB 需 40ms；静态展开成常规 MatMul/elementwise 后 10~18ms。unroll 与 `nn.LSTM` 在 fp64 下逐位等价（6.9e-15），无语义风险。
2. **stateful RNN 的 fp16 漂移主因是 vector 算子，不只是 cube cast**：cfio、hi/lo split 补偿都无效，因为状态更新 `c=f*c+i*g` 本身在跑 fp16；`origin` 让 vector 算子保 fp32 后漂移立即有界（cos_min 0.99996）。
3. **origin 的可用性取决于图里有没有融合算子**：fused LSTM 图上 origin 编译失败（内部 cube 要求 fp32）；unrolled 图全是常规算子，origin 可编译，cube 自动降 fp16。
4. **短门控不等于长流水可用**：FB 16 帧门控 0.995 看似可用，256 帧跌到 0.80。stateful 模型必须做长序列漂移测试再下结论。
5. **pyACL 8.3 API 与旧示例差异大**（见 3.3 表），板端脚本先小步探测每个调用的返回形状再写主流程。
6. **方法论**：固定种子冻结 GT → 本地 ONNX 门控 → ATC → 板端回放门控 + 长序列漂移 + 时延，所有候选对同一 GT 比较；决策与结果落 `reports/`。
