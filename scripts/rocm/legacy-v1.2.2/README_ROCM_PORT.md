# YuE2 Music T8 · AMD Radeon (ROCm) 完整移植报告

> **这是把 [T8mars/Comfyui-YuE2-T8](https://github.com/T8mars/Comfyui-YuE2-T8)（原版仅支持 NVIDIA CUDA）
> 完整移植到 AMD Radeon / Windows 原生 ROCm 的全记录。**
> 实测环境：AMD Radeon RX 9070 XT 16GB（gfx1201 / RDNA4）· Windows 11 build 26200 · t8 v1.2.2
> 四项能力（出歌 / 音频转谱 / 乐谱渲染 / 参考音色）**全部实测产出真实产物**，不只是 capability 标志。

---

## 0. TL;DR

| | NVIDIA（原版） | AMD（本移植） |
|---|---|---|
| PyTorch | `download.pytorch.org/whl/cu128` | `rocm.nightlies.amd.com/v2/gfx120X-all/`（AMD TheRock） |
| 应用代码改动 | — | **5 处**（由一个幂等脚本应用，见 §3） |
| 安装脚本改动 | — | **1 处**（轮子索引，见 §3） |
| 出歌 | ✅ | ✅ 175s 音频 / 613.7s |
| 音频转谱（SheetSage2+MERT） | ✅ | ✅ 24s 音频 / 36s |
| 乐谱渲染（playwright+abcjs） | ✅ | ✅ 产出 PDF + 钢琴试奏 WAV |
| 参考音色（Seed-VC+Demucs） | ✅ | ✅ 47s，产出 audio.flac |

**核心结论**：t8 出歌路径上唯一"长得像 CUDA"的应用代码只有
`torch.cuda.is_available()` 和 `torch.cuda.is_bf16_supported()` 两处自检 ——
ROCm 下 `torch.cuda` 就是 HIP 别名，两者都返回 True（报错文案里的"NVIDIA"只是文字）。
真正 NVIDIA-only 的只有**安装脚本里的轮子索引**，不是应用本身。

模型权重与 t8 的 pin **逐位一致**（YuE2-3B sha256 `1d55c42c…a59e9`、
YuE2-Vae `807ce9d5…7751346`；从 `m-a-p` 官方下载的字节与 `mrfakename` 镜像相同）。

---

## 1. 结果实测（可直接引用的数据）

| 验证 | 耗时 | 产物 |
|---|---:|---|
| 出歌（中文，175s 音频，seed 20260917） | 613.7s | `audio.flac` + `score.abc` |
| 音频转谱（24s 音频） | 36s | ABC 310 字符 + MIDI |
| 乐谱渲染 | 含上 | **PDF** + 钢琴试奏 `piano_mix.wav` |
| 参考音色（24s，diffusion_steps=8） | 47s | `audio.flac`（转换人声+伴奏重混），Seed-VC RTF 0.79 |
| doctor 自检 | 12s | GPU/BF16/四模型 SHA256 全过 |

性能参考（同一张卡）：
- 出歌速度：每 1 秒音频 ≈ 4.5 秒（eager 后端）
- **歌长 = 0.04 秒/token**（25 token/秒音频），默认 `semantic.max_tokens=9000` → 上限 6 分钟；
  歌长由**歌词段数**决定（官方示例只有 2 段所以只出 1 分钟）
- VAE 分块对速度影响巨大（同一 60s latent）：512 帧 → 101.6s；1024 帧 → 230.9s；不分块 → 285.0s
- AR 吞吐：eager 24.7 tok/s；CUDA graph 在短序列 46.3 tok/s、长序列反而 14.3 tok/s
  （掩码 SDPA 每步对全 capacity 算注意力，上游靠 `seqused_k` 规避，而 ROCm 拒绝该参数）

**一个正确性信号**：同 seed 同歌词，经「官方 CLI 直跑」与「t8 的 service→worker→vendored yue2」
两条完全不同的代码路径，产出音频时长**逐位一致**（`174.91866666666667s`）——移植未引入行为偏差。

---

## 2. 移植架构（为什么可行）

```
t8 的作业链路（每个作业起独立 worker 进程，隔离运行时）：
  service (runtime/core)  ──┬── kind=generate/plan/decode → core_worker      (runtime/core)
                            ├── kind=transcribe         → transcribe_worker  (runtime/transcribe)
                            ├── kind=voice_convert      → voice_worker       (runtime/voice)
                            └── kind=reference_cover    → workflow_worker    (core)

三个 worker 是三套互不相干的 Python 环境 → 可以分别处理各自的 GPU 兼容性问题。
```

关键事实（决定了移植的形状）：
- 出歌路径上唯一 CUDA 形状的代码就是上面两行自检；**没有 sm_XX / capability / nvml 检查**
- t8 的 `vendor/yue2/`、`vendor/seed-vc/` 两个模型栈**互相零引用**：
  音色转换是 t8 作者加的**通用后处理**（YuE2 出歌 → Demucs 分离 → Seed-VC 换音色 → 重混），
  与 YuE2 模型本身无关（官方文档明确说明 YuE2 **不保证**编辑后音色一致）
- 模型清单（`MODEL_MANIFEST.json` / `VOICE_MODEL_MANIFEST.json`）自带 path/size/sha256，
  校验脚本只认清单 —— 所以权重必须与 pin 逐位一致

---

## 3. 全部改动

### 3.1 源码 patch（7 处，由 `scripts/rocm/apply_rocm_port.py` 幂等应用）

| # | 文件 | 问题（现象 → 根因） | 修法 |
|---|---|---|---|
| 1 | `vendor/yue2/cuda_graph.py` | 解码时 `RuntimeError: [ROCm] mha_varlen_fwd: seqused_k must be nullopt`。GraphAR 用**只看 ATen schema** 的探测选后端；ROCm 下 `device.type=="cuda"` 为真且 schema 跨后端共享 → 误选 flash | HIP 上强制走掩码 SDPA。不能改传 `seqused_k=None`（变长 FA 会注意未使用的 cache 槽位 → 结果错误）。实测与 eager 数值差 0.0 |
| 2 | `app/yue2_app/core_worker.py` | `vae_core_frames` 由 budget 单值推导（`>12GiB → 1024`），隐含 24GB 卡。16GB A 卡上 1024 帧命中慢得多的 MIOpen solver：**VAE 313s vs 156s** | 按实卡显存自适应（<20GiB 用 512），请求可显式覆盖 |
| 3 | `runtime/voice/…/audiotools/ml/decorators.py` | demucs→dac→audiotools 在**类体导入时**求值 `dist.ReduceOp`；torch≥2.9 的 `torch.distributed` 是惰性模块，未初始化进程组前没有它 → `AttributeError`，Seed-VC 无法启动 | 未初始化时注入哨兵 ReduceOp |
| 4 | `vendor/seed-vc/inference.py` | CAMPPlus 与 RMVPE 是 `load_models()` 里仅剩的 fp32 模型；fp32 batchnorm 触发 MIOpen JIT → `miopenStatusUnknownError`；且调用点喂 fp32 特征 | 转 fp16（与 whisper/hubert/wav2vec 一致）+ 调用点转换 dtype |
| 5 | `app/yue2_app/voice_worker.py` | ① MIOpen 多个 fp32 kernel（spatial BN、RNN/GRU）都要 HIPRTC **运行时编译**，而 TheRock wheel 无 libc++ 头 → `type_traits file not found`；② torchaudio 2.11 的 `.save()` 走 torchcodec，Windows 无 FFmpeg 共享库 | ① 该 worker `torch.backends.cudnn.enabled=False`（ROCm PyTorch 里这就是 MIOpen 后端），conv/BN/RNN 走 PyTorch 原生实现；② `torchaudio.save` 替换为 soundfile |
| 6 | `app/yue2_app/updater.py` | 应用内「更新到 v1.3.0」会从 GitHub 拉发布包**整个覆盖 kit** → 5 处 ROCm patch 与 ROCm PyTorch 全被冲掉 | `public_update()` 恒返回"已是最新"；升级改为手动（重装新版后重跑本补丁脚本） |
| 7 | `app/web/index.html` + 3 个 worker 的自检文案 | 界面写"所有推理都在你的 **NVIDIA** GPU 上完成"，自检报错写"未检测到 **NVIDIA CUDA**"——在 HIP 构建上是误导（用户会以为不支持） | 改为 AMD / "CUDA/HIP"。**纯文案，无行为改动**（自检本身走 HIP 别名，本来就能过） |

> Patch 5 是**根治**：与其逐个 kernel 打地鼠（BN 修好炸 GRU），不如在该 worker 里绕开 MIOpen 的
> JIT 路径。代价仅是 Seed-VC/Demucs 内部部分 conv 吞吐（RTF 0.79 仍快于实时）；
> **出歌（core_worker）不受影响，继续用 MIOpen 跑重度 GEMM**。
> 也试过把 LLVM libc++ 头（1721 文件）平铺进 clang 资源目录 `lib/clang/23/include` ——
> **HIPRTC 的搜索路径不含资源目录，无效**。这条弯路记下来避免重复。

### 3.2 安装脚本（1 处）

`scripts/setup_rocm_runtime.ps1`（新增，替代上游 `setup.ps1` 的运行时部分）：
- 嵌入式解释器布局与上游完全一致（`runtime/<role>/python.exe` 在根；`python -m venv` 给不出这个布局，
  junction/复制 python.exe 都试过并失败 —— `sys.prefix` 解析错误）
- 轮子索引换成 AMD TheRock：`rocm[libraries,device-gfx1201]`（v4/whl）+ torch（v2/gfx120X-all）
- **用 uv 而非 pip**：AMD `rocm` 包是 sdist，embeddable 解释器无 setuptools，pip 报
  `Cannot import 'setuptools.build_meta'`；uv 为 sdist 建隔离构建环境
- 安装末尾断言 `torch.version.cuda is None`：否则 ROCm torch 装失败时 pip 会**静默**换上 CUDA 版
- python.org 从部分网络不可达且无超时会挂死 → 镜像列表（华为云/npmmirror/阿里）+ curl 超时

### 3.3 三个角色统一同一个 torch 构建（重要偏离，写明理由）

**上游 pin `torch==2.8.0 torchaudio==2.8.0` 在 ROCm 索引上无法解析**
（没有稳定的 cp311 torchaudio 2.8.0，只有 `2.8.0a0` 预发布且配的是旧 ROCm）。
且实测踩到：uv 会**独立解析**两者，配出 `torch 2.9.0+rocm7.10.0a20251117` ×
`torchaudio 2.9.0+rocm7.13.0a20260416` 的 ABI 不匹配组合；更糟的是 torch 2.9.0（2025-11 构建）
需要 `hipsparselt`，而它配对的 rocm 7.10 SDK 和整个 v4 索引里**都没有**这个库。

**解法**：三个角色统一用 core 已验证的 `torch==2.11.0+rocm7.13.0a20260416`
（该 rocm 构建的 SDK 列出 22 个库、含 hipsparselt；torch/torchaudio cp311 都有这个构建）。

---

### 3.4 移植脚本的两条工程纪律（踩过才知道）

1. **补丁脚本必须保留原文件的换行符。** 早期版本用 Python `Path.write_text()` 写回，
   在 Windows 上会把每个 `\n` 翻译成 `\r\n` —— 一个 2 行的修复变成整文件 diff
   （实测：`transcribe_worker.py` 从 4836 B 涨到 4957 B，多出的 106 字节 = 106 个多余 CR）。
   现在统一用 `open(..., newline="")` 读写，行尾原样保留。
2. **整套改动必须可重放且幂等。** `apply_rocm_port.py` 在**未打补丁的上游检出**上实测：
   8 条锚点全部 `PATCH`、`APPLY_EXIT=0`；重跑全部 `SKIP`；且产出与本文档描述的
   已实测 kit **逐字节一致**（7 个源文件 SHA256 相同）。这是提 PR 的前提。

### 3.5 常见参数问答

**「高级设置里的显存预算默认 23.5，要不要改成 15.5？」——不用改。**

`memory_budget_gib` 不是实际占用，而是**进程上限**，且会被 pipeline 按实卡自动夹取：

```python
budget = min((memory_budget_gib - 2) * 2**30, total - 2 * 2**30)
# 16 GB 卡（15.92 GiB）上：23.5 -> min(21.5, 13.92) = 13.92 GiB
#                          15.5 -> min(13.5, 13.92) = 13.5  GiB  ← 反而更低，零收益
```

实测峰值只用 11~12 GB。真正影响速度的是 `vae_core_frames`（1024 帧比 512 帧慢 2.3×），
而 Patch 2 已把它改成**按实卡显存自动选 512**，与这个预算值无关。所以保持默认即可。

---

## 4. 复现步骤（全新机器）

```
0) 前置：AMD 驱动 + Windows 11；磁盘约 45GB；网络能访问 hf-mirror.com 与 rocm.nightlies.amd.com
   （python.org 可能不可达 —— 脚本带镜像回退；代理可选，仅 GitHub 拉取时需要）

1) 克隆 t8（代码即可，模型/运行时不用）：
   git clone --depth 1 https://github.com/T8mars/Comfyui-YuE2-T8.git YuE2-T8

2) 安装三个 ROCm 运行时（每个 6~7GB）：
   powershell -ExecutionPolicy Bypass -File scripts\setup_rocm_runtime.ps1 -Role core
   powershell -ExecutionPolicy Bypass -File scripts\setup_rocm_runtime.ps1 -Role transcribe
   powershell -ExecutionPolicy Bypass -File scripts\setup_rocm_runtime.ps1 -Role voice

3) 应用全部源码 patch：
   python scripts\rocm\apply_rocm_port.py <kit目录>

4) 模型（12.2GB，经 hf-mirror 分块续传）：
   python scripts\rocm\fetch_pinned_models.py  # SheetSage2 + MERT 补全 + MODEL_MANIFEST.json
   python scripts\rocm\fetch_voice_models.py   # Seed-VC + Demucs + VOICE_MODEL_MANIFEST.json
   # 完整性用 t8 自带脚本校验：
   runtime\core\python.exe scripts\verify_models.py --root <kit目录>
   runtime\core\python.exe scripts\verify_voice_models.py --root <kit目录>

5) 启动：双击 start_rocm.bat  →  http://127.0.0.1:8189

6) 验证（t8 自带的 doctor + 三类作业）：
   python scripts\rocm\verify_capabilities.py
```

注意：`scripts\rocm\fetch_*.py` 是**镜像直连**的分块续传实现 —— huggingface_hub 客户端在此
网络环境下不可用（hub 对 `huggingface.co` 大文件 `RemoteDisconnected`、镜像与 hub 客户端
不兼容、Windows 无开发者模式时缓存符号链接 `PermissionError`）。

---

## 5. 已知限制 / 未解项

1. **VAE 分块提速未复测**：Patch 2 已应用，但实测的 613.7s 那次跑在补丁之前
   （当时 `vae_core_frames=1024`）。预期复测 613.7s → ~457s。worker 每次作业新起进程，无需重启服务。
2. **MIOpen 的 JIT 依赖未根治**：Patch 5 只让 voice worker 绕开了它。core 的 GEMM 走的是
   预编译 kernel，实测无碍；但若未来 core 路径触发新的 JIT kernel，会以同样方式失败。
   彻底解法是给 comgr 提供 libc++ 头（资源目录平铺无效，正确姿势待查）。
3. **`quantization="fp8"` 绝不可用**：capability 门会通过（capability `(12,0) >= (8,9)`），
   `torch._scaled_mm` 也存在，但对反量化权重的相对误差达 90~117 —— **不是精度损失，是计算错误**，
   会静默产出垃圾音频。
4. **torchaudio 2.11 的 `.save()` 依赖 torchcodec**：本移植在 voice worker 内替换为 soundfile
   （仅写 WAV，等价）。若要通用的 torchcodec，需自备 FFmpeg 共享库（PyAV 的 DLL 实测未能配对）。
5. **上游 `setup.ps1` 未改动**：本移植用新脚本并排存在，未删改上游文件 —— 便于 rebase 与提 PR。

---

## 6. 常见坑速查（全部实测踩过）

| 坑 | 现象 | 解法 |
|---|---|---|
| python.org 被墙 | curl 无超时永久挂起 | `--connect-timeout/--max-time` + 镜像回退；或预置归档 |
| pip 装不了 AMD `rocm` | `Cannot import 'setuptools.build_meta'` | 用 uv（sdist 隔离构建） |
| pip 静默换 CUDA torch | ROCm torch 失败后 pip 去 PyPI 拉 cu 版 | 断言 `torch.version.cuda is None` |
| venv 布局不匹配 | t8 要 `runtime/<role>/python.exe` 在根 | 必须用 embeddable；junction/复制 python.exe 均失败 |
| hub 客户端下不了模型 | `RemoteDisconnected` / `LocalEntryNotFoundError` / 符号链接 `PermissionError` | HTTP 分块续传直连镜像 |
| MIOpen 需要写权限 | `miopenStatusInternalError`（SQLite 调优库） | 沙箱/只读环境会撞；正常用户权限无碍 |
| 首次 VAE 解码异常慢 | 首跑 69.7s/chunk vs 之后 41~55s | MIOpen 一次性 solver 调优，非配置问题 |
| 删 junction | `Remove-Item -Recurse` 可能删目标内容 | `cmd /c rmdir` |
| SheetSage2 加载失败 | 缺 `configuration_sheetsage2.py` 等 | trust_remote_code 模型要下**全部** .py，不能只下 REQUIRED_FILES |
| torchaudio.save 失败 | `No module named torchcodec` | 见 Patch 5/6 |
| torch/torchaudio 配错对 | ABI 不匹配 / 缺 hipsparselt | 钉同一完整构建串（§3.3） |

---

## 7. 目录结构

```
D:\YuE2\T8\start_rocm.bat            双击启动（ROCm 环境变量）
D:\YuE2\T8\runtime\core\             py3.12.10 + ROCm torch 2.11（出歌）
D:\YuE2\T8\runtime\transcribe\       py3.11.9  + ROCm torch/torchaudio 2.11（转谱）
D:\YuE2\T8\runtime\voice\            py3.11.9  + ROCm torch/torchaudio 2.11（音色）
D:\YuE2\models\                      12.2GB，四模型 + 两个 manifest
D:\YuE2\venv\                        官方 YuE2 CLI 路线（run_yue2.bat，与本移植独立）
D:\YuE2\T8\scripts\rocm\             本移植的全部脚本
```

许可证：t8 代码 Apache/MIT（见其 LICENSE）；YuE2 权重 CC BY-NC 4.0（**非商用**）；
SheetSage2/MERT 见各自 LICENSE；Seed-VC GPL-3.0。发布前请核对。
