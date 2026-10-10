# 给 t8 上游提 PR 的工作文档

> 目标会话：**对接 GitHub 的会话**用本文档。
> 上游仓库：https://github.com/T8mars/Comfyui-YuE2-T8
> 本移植基于：**v1.2.2**（commit `a9cc3af081ddf8025f75c4993f128a4554dafe31`，2026-09-12）
> 本地参考 checkout：`D:\DSHWEB\_yue_probe\t8\`

---

## 0. 提 PR 前的三个判断（先做，别跳过）

1. **先开 issue 沟通，不要直接甩 PR。** 理由：
   - Patch 1/3/4/5 是 **bug 修复**（在 NVIDIA + torch 2.11 上同样成立，见 §2.1），
     作者大概率愿意收；
   - Patch 2 是**行为变化**（16GB 卡默认 VAE 分块从 1024 改 512），需要作者认可"按显存自适应"这个方向；
   - ROCm 支持是否进上游，取决于作者是否愿意在 CI/文档里引入 AMD 分支 —— 这是维护成本问题。
2. **拆分 PR**：建议 3 个独立 PR，不要合一个：
   - PR-A「fix: torch 2.10+ 兼容」（Patch 1/3/4/5，纯 bug 修复，CUDA 用户也受益）
   - PR-B「feat: device-aware VAE tile size」（Patch 2）
   - PR-C「docs+scripts: AMD ROCm port guide」（新脚本 + 文档，不改上游代码）
3. **PR-A 必须用 CUDA 论证**：这些 bug 在 NVIDIA 上装 torch 2.10/2.11 也复现
   （是我们本地理论推断 + 代码事实，**没有 NVIDIA 机器实测**——发 issue 时要如实说明，
   请维护者/社区帮忙确认，避免被当成"AMD 专属问题"而拒绝）。

---

## 1. PR-A：fix: torch 2.10+ compatibility (4 issues, all reproducible on NVIDIA too)

> 主题：t8 的 requirements 与 vendored 代码假设 torch 2.8；升级到 torch≥2.9 会踩 4 个问题。
> 我们在 ROCm/torch 2.11 上全部实测命中。以下每个都给「现象 / 根因 / 修法」。

### 1.1 `descript-audiotools` 在 import 时崩溃（阻断 Seed-VC 启动）

```
AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'
  audiotools/ml/decorators.py:288  op: dist.ReduceOp = dist.ReduceOp.AVG
```
- **根因**：demucs → dac → descript-audiotools 在 `Tracker` 类体里对 `dist.ReduceOp`
  求值；torch≥2.9 把 `torch.distributed` 改成惰性模块，未初始化进程组前不暴露 `ReduceOp`。
- **影响**：`voice_convert` / `reference_cover` 完全无法启动。
- **修法**：import 时若 `dist` 无 `ReduceOp`，注入哨兵（Seed-VC 从不初始化分布式）。
- 文件：`app/yue2_app` 之外，实际在 **voice 运行时的 site-packages**（audiotools 是 demucs 的
  传递依赖）→ 上游的正确修法可能是 **pin `descript-audiotools` 的一个兼容版本，或在
  voice requirements 里加补丁说明**。这需要作者定夺，所以更要先开 issue。

### 1.2 Seed-VC 的 CAMPPlus / RMVPE 在 MIOpen 上崩（fp32 batchnorm JIT）

```
MIOpen(HIP): Error [Compile] MIOpenBatchNormFwdInferSpatial.cpp
  fatal error: 'type_traits' file not found   →  miopenStatusUnknownError
```
- **根因**：MIOpen 对部分 fp32 kernel 需要 HIPRTC 运行时编译 C++；TheRock 的 wheel
  不带 libc++ 头，comgr 编译失败。
- **修法（我们采用的）**：CAMPPlus/RMVPE 转 fp16（`load_models()` 里其它模型本来就
  `.half()`），fp16 走不同 kernel、无需 JIT。实测通过（RTF 0.79）。
- **对 NVIDIA 用户的相关性**：若 TheRock 之外的 MIOpen 问题不存在，这条纯粹是 AMD 的；
  但 `load_models()` 里模型精度不一致（CAMPPlus 独自 fp32）本身值得统一。

### 1.3 `torchaudio 2.11` 的 `.save()` 需要 torchcodec

```
ImportError: TorchCodec is required for save_with_torchcodec
```
- **根因**：torchaudio 2.11 把 `.save()` 改走 torchcodec；Windows wheel 不带 FFmpeg 共享库。
- **修法（我们采用的）**：在 voice worker 内把 `torchaudio.save` 替换为 soundfile
  （Seed-VC 只写 WAV，等价）。
- **上游更优解**：requirements 加 `torchcodec` 并在文档写明 FFmpeg 共享库需求；
  或作者把 Seed-VC 的保存改成 soundfile。

### 1.4 `cuda_graph.py` 对 torch 2.10+ 的假设

- `vendor/yue2/cuda_graph.py` 的注释明确写 "Torch 2.10 is pinned by the package"，
  并对 `torch.ops.aten._flash_attention_forward` 的 schema 做**字符串探测**来选后端。
- **schema 是跨后端共享的**：任何"schema 里有 X 但该后端实现不支持 X"的组合都会误判
  （我们就是在 ROCm 上命中：schema 有 `seqused_k`，ROCm 的 `mha_varlen_fwd` 拒绝它）。
- **修法**：探测后**再实测一次**（try/except 或 capability 查询），而不是只信 schema。
  这对 torch 升级后的 CUDA 用户同样有价值。

---

## 2. PR-B：feat: device-aware VAE tile size

> `app/yue2_app/core_worker.py` 的 `create_pipe()` 没有把 `vae_core_frames` 透传给
> pipeline，而 pipeline 内部按 `memory_budget_gib` 单值推导（`512 if <=12 else 1024`），
> 隐含 24GB 卡假设。

**实测收益（同一请求 `zh_song.json`，seed 20260917，产出 174.919 s 音频）**：

| | wall | VAE 解码 | `vae_core_frames` |
|---|---:|---:|---|
| 改前 | 613.7 s | 313.0 s | 1024（由 `memory_budget_gib=23.5` 推导） |
| 改后 | **393.4 s** | **107.0 s** | 512（按实卡 15.92 GiB 自适应） |
| 收益 | **−220.3 s（−35.9%）** | −206 s | — |

折算：**3.51 s/音频秒 → 2.25 s/音频秒**。改后甚至快于官方 CLI 直跑同请求的 423.4 s。

**建议修法**：`create_pipe` 增加
```python
vae_core_frames=int(request.get("vae_core_frames") or _default_vae_frames())
```
其中 `_default_vae_frames()` 按实卡显存选（<20GiB → 512，否则维持上游 1024）。
对 24GB NVIDIA 用户是零变化；对 16GB 用户是 36% 提速；API/前端可选择性透传。

**顺带说明**：`memory_budget_gib` 本身不需要用户调 —— 它会被 pipeline 自动夹到
`min(budget-2, total-2)`，16GB 卡上填 23.5 实际就是 13.92 GiB。真正影响速度的只有分块大小。

---

## 3. PR-C：docs/scripts: AMD Radeon (ROCm) port guide

- 新增 `scripts/rocm/*` 与 `docs/ROCM_PORT.md`（内容取自 `README_ROCM_PORT.md`）：
  - 参数化运行时安装器（core/transcribe/voice），把 cu128 索引换成 AMD TheRock
  - `apply_rocm_port.py` 幂等应用全部 patch
  - 镜像直连的模型下载器（huggingface_hub 在部分网络不可用）
- **不改动上游任何现有文件**（上游 setup.ps1 保持原样，ROCm 用户用新脚本并排安装）
- 文档里写明：模型哈希与上游 pin 逐位一致；四项能力在 RX 9070 XT 上实测通过

### 3.1 另外两条建议一并提（都属于 PR-B/C 的性质）

1. **禁止应用内自更新覆盖手改环境**（我们的 Patch 6）。t8 的「更新到 vX」会整包重解压，
   对任何本地改过 kit 的用户（不只是 AMD）都会静默冲掉改动。建议：
   升级前检测 `runtime/` 与源码是否有本地修改，或至少在文档里明确警告。
   我们在移植版里让它恒返回"已是最新"。
2. **设备文案不该写死 NVIDIA**（我们的 Patch 7）。`index.html` 写
   "所有推理都在你的 NVIDIA GPU 上完成"，三个 worker 的自检报错写
   "未检测到 NVIDIA CUDA"。在 HIP 构建上这些文案是误导（用户会以为硬件不支持，
   而实际上自检走 `torch.cuda` 别名、HIP 下本来就通过）。建议改成中性
   （"本地 GPU" / "CUDA/HIP"）或按 `torch.version.hip` 动态显示。

### 3.2 给维护者的工程提示（我们踩到的）

- **补丁脚本要保留原文件换行符**：Python `Path.write_text()` 在 Windows 会把 `\n`
  变 `\r\n`，一个 2 行修复会变成整文件 diff。我们的脚本统一用 `open(..., newline="")`。
- **给 vendored 代码打补丁时注意多锚点**：同一文件可能有多处改动，幂等标记应选
  "全部改动完成后才出现"的特征串，否则会漏打后续锚点（我们在 `inference.py` 上踩过）。

---

## 4. 提交材料清单（GitHub 会话需要准备的）

| 项 | 来源 | 状态 |
|---|---|---|
| 移植报告（中文） | `D:\DSHWEB\_yue_probe\README_ROCM_PORT.md` | ✅ 已写 |
| patch 应用脚本 | `D:\YuE2\T8\scripts\rocm\apply_rocm_port.py` | ✅ 已在 kit 内 |
| 运行时安装器 | `D:\YuE2\T8\scripts\setup_rocm_runtime.ps1` | ✅ 已在 kit 内 |
| 验证脚本 | `D:\YuE2\T8\scripts\rocm\verify_capabilities.py` | ✅ |
| 性能/正确性数据 | `D:\YuE2\PROGRESS.md` §3 | ✅ |
| **复测 Patch 2 的提速数据** | 已完成：613.7 s → **393.4 s**（−35.9%），见 §2 | ✅ |
| **每个 patch 的最小 diff 文件**（.patch/.diff 格式） | 由 apply_rocm_port.py 的锚点生成 | ⬜ |
| NVIDIA 复现佐证（1.1/1.3 两条） | 请社区/作者确认 | ⬜ |

> 注意：`D:\YuE2\T8` 是**已打补丁**的工作副本。给上游提 PR 时，diff 的 base 应是
> 上游 v1.2.2（`a9cc3af…`），只包含上面列的 5 处源码改动 —— 不要把 `runtime/`、
> `models/`、`outputs/`、`cache/`、`logs/` 等运行产物带进 diff。

---

## 5. 语气与署名建议

- 开头致谢作者：t8 的架构（隔离 worker、vendored 依赖、SHA 清单校验）是这次移植
  能低成本完成的关键 —— 三个运行时可以独立替换，模型完整性校验让"权重逐位一致"可证。
- 明确标注非商用：YuE2 权重 CC BY-NC 4.0。
- 附上机器信息（RX 9070 XT / gfx1201 / Windows 11 / ROCm 7.13 wheel）与我们实测的四项数据。
