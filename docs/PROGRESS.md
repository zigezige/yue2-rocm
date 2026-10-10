# YuE2 Music T8 · AMD ROCm 完整移植 ✅

> 更新：②③ 全部完成并实测验证。**T8 四项能力在 RX 9070 XT 上全部实测可用。**

---

## 1. 最终状态

| 能力 | 状态 | 实测结果 |
|---|---|---|
| **generation** 出歌 | ✅ | 175s 音频，613.7s 完成（含可优化的 VAE 分块） |
| **transcription** 音频转谱 | ✅ | 24s 音频 → ABC 乐谱，36s 完成 |
| **score_renderer** 乐谱渲染 | ✅ | playwright/abcjs 产出 **PDF 乐谱** + 钢琴试奏 WAV |
| **voice_conversion** 参考音色 | ✅ | Demucs 分离 + Seed-VC 转换 + 重混，47s 完成，`audio.flac` |

四项能力全部**实测跑通**（不是只看 capability 标志）。UI 横幅显示「运行环境已就绪」。

服务：`D:\YuE2\T8\start_rocm.bat` → `http://127.0.0.1:8189`

---

## 2. 移植改动清单（对 t8 的全部改动）

一个幂等脚本应用全部源码 patch：`scripts/rocm/apply_rocm_port.py`

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| 1 | `vendor/yue2/cuda_graph.py` | 解码时 attention 后端误判。ROCm 下 `device.type=="cuda"` 为真、ATen schema 跨后端共享 → 选中 flash → `RuntimeError: [ROCm] mha_varlen_fwd: seqused_k must be nullopt` | HIP 上强制掩码 SDPA（与 eager 数值差 0.0） |
| 2 | `app/yue2_app/core_worker.py` | `vae_core_frames` 由 budget 单值推导（>12GiB 用 1024），隐含 24GB 卡假设。16GB A 卡上 1024 帧命中极慢的 MIOpen solver：VAE 313s vs 156s | 按实卡显存自适应（<20GiB 用 512），支持请求显式覆盖 |
| 3 | `runtime/voice/.../audiotools/ml/decorators.py` | demucs→dac→audiotools 在**类体导入时**求值 `dist.ReduceOp`；torch≥2.9 的 `torch.distributed` 是惰性模块，未初始化时没有 `ReduceOp` → `AttributeError`，Seed-VC 无法启动 | 无分布式时注入哨兵 `ReduceOp` |
| 4 | `vendor/seed-vc/inference.py` | CAMPPlus 与 RMVPE 是 `load_models()` 里**仅剩的 fp32 模型**（其它都 `.half()`），fp32 batchnorm 触发 MIOpen JIT → 失败；且调用点喂 fp32 特征 | 转成 fp16（与其它模型一致）+ 调用点转换 dtype |
| 5 | `app/yue2_app/voice_worker.py` | ① MIOpen 多个 fp32 kernel（spatial BN、GRU/RNN）都要 HIPRTC JIT，而 TheRock wheel 无 libc++ 头 → `type_traits file not found`；② torchaudio 2.11 的 `.save()` 走 torchcodec，Windows 无 FFmpeg 共享库 | ① 该 worker 关闭 MIOpen（`torch.backends.cudnn.enabled=False`），conv/BN/RNN 走 PyTorch 原生实现；② `torchaudio.save` 替换为 soundfile |

新增（不改上游）：
```
scripts/setup_rocm_runtime.ps1     参数化运行时安装器（core/transcribe/voice），ROCm 索引
scripts/rocm/apply_rocm_port.py    幂等应用全部 5 处 patch
scripts/rocm/fetch_pinned_models.py / fetch_voice_models.py / stage_bootstrap.py
start_rocm.bat                     启动器（ROCm 环境变量）
runtime/ffmpeg/*.dll               PyAV 借来的 FFmpeg 共享库（备用）
clang 资源目录里的 libc++ 头        供 HIPRTC JIT 使用（虽然最终用方案5绕开了）
```

---

## 3. 关键结论（发布素材）

### 3.1 为什么 t8 在 A 卡上「只差两步」

出歌路径上唯一 CUDA 形状的代码是：
```python
if not torch.cuda.is_available(): raise ...      # ROCm 下为真（HIP 别名）
if not torch.cuda.is_bf16_supported(): raise ... # True（gfx1201 支持 BF16）
```
真正不可移植的只有 **cu128 轮子索引** —— 那是安装脚本的选择。模型哈希与 t8 的 pin
**完全一致**（YuE2-3B `1d55c42c…a59e9`，YuE2-Vae `807ce9d5…7751346`，从 `m-a-p` 下载的字节
与 `mrfakename` 镜像逐位相同）。

### 3.2 torch/torchaudio 版本（最大的坑）

- uv 会**独立解析**两者，配出 `torch 2.9.0+rocm7.10` × `torchaudio 2.9.0+rocm7.13` 的错误组合
- torch 2.9.0（2025-11 构建）需要 `hipsparselt`，但它配对的 rocm 7.10 SDK 与整个 v4 索引里**都没有**
- **解法：三个角色统一用 core 已验证的 `2.11.0+rocm7.13.0a20260416`**（SDK 22 库含 hipsparselt，
  torch/torchaudio cp311 都有此构建）
- 上游 pin 的 `torch==2.8.0 torchaudio==2.8.0` 在 ROCm 索引里**无法解析**（无稳定 2.8.0 torchaudio）

### 3.3 MIOpen JIT 与 libc++（最深的一个坑）

TheRock 的 PyTorch wheel **不带 libc++（C++ 标准库）头文件**。MIOpen 在多个 fp32 kernel
上需要 HIPRTC **运行时编译**（spatial batchnorm、RNN/GRU），comgr 找不到 `type_traits` →
```
fatal error: 'type_traits' file not found  →  miopenStatusUnknownError
```
fp16 走完全不同的 kernel，不需要 JIT —— 所以 half 化模型能绕开单点。
**根治方案**：voice worker 里 `torch.backends.cudnn.enabled = False`（ROCm PyTorch 中
cudnn 后端就是 MIOpen），conv/BN/RNN 走 PyTorch 原生实现。
clang 资源头文件目录（`lib/clang/23/include`）**平铺**进 libc++ 头（1721 个文件）也试过，
但 HIPRTC 的搜索路径不含资源目录，**无效**——这条弯路已记录。

### 3.4 实测性能

| 指标 | 数值 |
|---|---|
| 出歌速度（eager） | 每 1 秒音频 ≈ 4.5 秒 |
| 时长与 token | **0.04 秒/token**（25 tok/s 音频），默认上限 6 分钟 |
| VAE 分块（60s latent） | 512→101.6s / 1024→230.9s / full→285.0s |
| AR 吞吐 | eager 24.7 tok/s；CUDA graph 短序列 46.3 但长序列 14.3 |
| Seed-VC RTF | 0.79（比实时快） |
| 音色转换全程 | 47s（24s 音频，含 Demucs 分离） |
| 转谱全程 | 36s（24s 音频，含 PDF 渲染） |

### 3.5 其它已验证的坑（详见 DEPLOY_NOTES.md）

- python.org 被墙 → 镜像 + curl 超时 + 预置归档
- pip 装不了 AMD `rocm` sdist（无 setuptools）→ **用 uv**
- ROCm torch 装失败后 pip 会静默换 CUDA torch → 安装脚本断言 `torch.version.cuda is None`
- `python -m venv` 布局 ≠ t8 的 embeddable 布局（junction/拷贝都不行）
- huggingface_hub 与镜像不兼容 + Windows 符号链接 → HTTP 分块续传（6 路并发 60 MiB/s）
- SheetSage2/MERT 需要**全部**远程代码 .py（trust_remote_code），只下 REQUIRED_FILES 会加载失败
- MIOpen 需要可写的 SQLite 调优库（沙箱/只读环境会 `miopenStatusUnknownError`）
- 首次 VAE 解码异常慢是 MIOpen 一次性 solver 调优，不是配置问题
- **不要开 `quantization="fp8"`**：capability 门会通过但 `torch._scaled_mm` 结果错误
- 删 junction 用 `cmd /c rmdir`，`Remove-Item -Recurse` 可能删掉目标内容

---

## 4. 目录与运行

```
D:\YuE2\T8\start_rocm.bat      # 双击启动 T8 WebUI（ROCm 环境变量已配好）
D:\YuE2\T8\runtime\core\       # py3.12.10 + ROCm torch 2.11（出歌）
D:\YuE2\T8\runtime\transcribe\ # py3.11.9 + ROCm torch/torchaudio 2.11（转谱）
D:\YuE2\T8\runtime\voice\      # py3.11.9 + ROCm torch/torchaudio 2.11（音色）
D:\YuE2\models\                # 12.2 GB，四个模型 + 两个 manifest
D:\YuE2\venv\                  # 官方 CLI 路线（run_yue2.bat）
```

验证脚本：`D:\DSHWEB\_yue_probe\t8_verify_capabilities.py`（doctor + 转谱 + 音色转换）
提交出歌：`D:\DSHWEB\_yue_probe\t8_submit_job.py <request.json>`

磁盘：D: 剩约 47 GB。

---

## 5. 待办（可选）

1. ~~复测 Patch 2（vae_core_frames=512）的提速~~ ✅ **已完成**：
   同一中文请求（seed 20260917，174.919 s 音频）实测 **613.7 s → 393.4 s（−35.9%）**，
   VAE 解码 **313.0 s → 107.0 s**。Patch 后的 T8 路线已快于官方 CLI 直跑的 423.4 s。
   `offload_ar` True/False 只差 4%（噪声内），保持默认 True。详见 `README_ROCM_PORT.md` §1.1。
2. **发布**：整理 GitHub README（本文档 §2/§3 即素材）+ 提交 patch 到 t8 上游（PR）
3. **B站视频**：`tech-video-production` / `media-publish` skill 可用；
   素材：7 首歌（D:\YuE2\outputs）、转谱 PDF、音色转换结果、性能对比表
4. 中文歌已有 2:55 的 `zh_demo01` 与 T8 链路的 `20260912-221159-666c25ab`（同 seed 同结果）
