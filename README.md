# YuE2-T8-ROCm

**把 [T8mars/Comfyui-YuE2-T8](https://github.com/T8mars/Comfyui-YuE2-T8)（原版仅支持
NVIDIA CUDA）完整移植到 AMD Radeon / Windows 原生 ROCm 的可用整合包。**

实测环境：**AMD Radeon RX 9070 XT 16GB（gfx1201 / RDNA4）· Windows 11 build 26200 ·
原生 Windows ROCm（无 CUDA、无 Triton、无 flash-attn）**

四项能力**全部实测产出真实产物**（不是只看 capability 标志）：

| 能力 | 实测 | 耗时 |
|---|---|---|
| 出歌 | 175 s 中文歌 → `audio.flac` + `score.abc` | 393.4 s |
| 音频转谱（SheetSage2 + MERT） | 24 s 音频 → ABC + MIDI | 36 s |
| 乐谱渲染（playwright + abcjs） | **PDF 乐谱** + 钢琴试奏 WAV | 含上 |
| 参考音色（Seed-VC + Demucs） | 换音色人声 + 伴奏重混 `audio.flac` | 47 s（RTF 0.79） |

> **这个仓库不含任何模型权重。** `models/` 只有目录骨架 + 一份说明每个文件应有的大小与
> sha256 的 README，权重由安装器下载。见 [`models/README.md`](models/README.md)。

---

## 快速开始

前置：AMD 驱动 + Windows 11；磁盘约 45 GB；网络能访问 `hf-mirror.com` 与
`rocm.nightlies.amd.com`。

```powershell
git clone https://github.com/Newaiguy/YuE2-T8-ROCm.git
cd YuE2-T8-ROCm

# 1) 建 ROCm 运行时（嵌入式 CPython 3.12 + AMD TheRock torch，约 6-7 GB）
powershell -ExecutionPolicy Bypass -File scripts\rocm\setup_rocm_runtime.ps1

# 2) 模型（约 12.2 GB；huggingface.co 不通时用镜像脚本）
runtime\python.exe -m huggingface_hub.cli.hf download t8star/YuE2-Comfy `
    --revision a083f106499daead99259dd0c443a5494254cfc5 --local-dir models
#    或： runtime\python.exe scripts\rocm\fetch_mirror_models.py
#    （默认只下载约 12.2 GB 的核心模型；全量仓库需显式加 --all）

# 3) 校验（走 t8 自己的清单）
runtime\python.exe scripts\verify_models.py --root .
runtime\python.exe scripts\verify_voice_models.py --root .

# 4) 启动
start_rocm.bat            # 浏览器打开 http://127.0.0.1:8189

# 5) 端到端验证
python scripts\rocm\verify_capabilities.py
```

> 本仓库里的源码**已经打好全部移植补丁**，所以第 2 步之后不需要再跑
> `apply_rocm_port.py`（它会全部报 SKIP）。如果你是拿 upstream 的干净检出，
> 才需要 `python scripts\rocm\apply_rocm_port.py .`。

### RX 6800 XT / gfx1030 注意事项

RX 6800 XT 属于 `gfx1030`。AMD 的 TheRock `v4/whl` 索引现在提供了可配套安装的
Windows wheel：`torch 2.9.1+rocm7.15.0a20260715`、
`torchaudio 2.9.0+rocm7.15.0a20260715` 和
`amd-torch-device-gfx1030`。安装器默认锁定这组版本，并在下载前检查三个 wheel，
避免静默回退到 CUDA/CPU 版本。运行时仍需要较新的 AMD Windows 驱动；安装完成后先运行
`runtime\python.exe scripts\rocm\verify_rocm.py`，再启动 WebUI 和真实出歌测试。

---

## 移植改了什么

源码改动 **3 处**（都在最小范围内，且尽量做成设备无关）：

| # | 文件 | 问题 → 修法 |
|---|---|---|
| 1 | `vendor/yue2/cuda_graph.py` | 解码时 `RuntimeError: [ROCm] mha_varlen_fwd: seqused_k must be nullopt`。后端探测**只看 ATen schema**，而 schema 跨后端共享 → 在 HIP 上误选 flash。改为 HIP 构建强制走掩码 SDPA（实测与 eager 数值差 **0.0**） |
| 2 | `vendor/seed-vc/inference.py` | CAMPPlus / RMVPE 是 `load_models()` 里仅剩的 fp32 模型，fp32 batchnorm 触发 MIOpen 经 HIPRTC 运行时编译，而 TheRock wheel 不带 libc++ 头 → `'type_traits' file not found` → `miopenStatusUnknownError`。改为跟随 `fp16` |
| 3 | `app/yue2_app/voice_worker.py` | 同类 JIT 失败在别的 fp32 kernel 上复发。该 worker 关闭 MIOpen（ROCm PyTorch 里 `torch.backends.cudnn` 就是 MIOpen），conv/BN/RNN 走原生实现。**仅在 HIP 上生效，CUDA 用户保持 cuDNN** |
| 4 | `app/yue2_app/core_worker.py` | VAE 分块由 budget 单值推导（`>12GiB → 1024`），隐含 24 GB 卡。改为按实卡显存自适应（`<20GiB → 512`），**同一请求 613.7 s → 393.4 s（−35.9 %）** |

站点包改动 **1 处**（无法进仓库，故提供脚本）：`descript-audiotools` 在类体导入时对
`dist.ReduceOp` 求值，torch ≥ 2.9 把它变成惰性模块 → `AttributeError`，Seed-VC 完全
无法启动。见 `scripts/rocm/patch_audiotools.py`。

完整记录、性能数据、踩过的坑：**[`docs/ROCM_PORT.md`](docs/ROCM_PORT.md)**。
三运行时时期的原始报告见 [`docs/PORT_REPORT.md`](docs/PORT_REPORT.md)。

---

## 与上游的关系

本仓库基于上游 **`main`（v1.3.0, `c91ab46`）**，把移植内容整理成了三个上游 PR：

| 分支 | 内容 |
|---|---|
| `fix/rocm-torch-compat` | 上面 1–3 三处源码修复 |
| `feat/device-aware-vae-tile` | 上面第 4 处（按显存自适应 VAE 分块） |
| `docs/rocm-port-guide` | `docs/ROCM_PORT.md` + `scripts/rocm/*`（纯新增文件） |

这些分支在 fork [Newaiguy/Comfyui-YuE2-T8](https://github.com/Newaiguy/Comfyui-YuE2-T8) 上。
提 PR 的方案与论证见 [`docs/UPSTREAM_PR_PLAN.md`](docs/UPSTREAM_PR_PLAN.md)。

上游若合并了对应改动，本仓库可以用 `scripts/rocm/apply_rocm_port.py` 重新对齐
（它会跳过已经存在的改动，锚点消失的条目报 `GONE` 而不是失败）。

> ⚠️ `pyproject.toml` 里的节点名仍然是上游的 `yue2-t8`（ComfyUI 节点包名，必须保持一致才能
> 被 ComfyUI 正确加载）。**本仓库不发布、也不应发布到 ComfyUI Registry** ——
> 那里对应的是上游项目。上游的 `.github/workflows/publish.yml` 已从本仓库移除，
> 正是为了避免误发布。

---

## 已知限制

1. **`quantization="fp8"` 绝不可用。** capability 门会通过（本机报 `(12, 0)`），
   `torch._scaled_mm` 也存在，但对反量化权重的**相对误差 90–117** —— 不是精度损失，
   是**计算错误**，会静默产出垃圾音频。
2. **MIOpen 的 JIT 依赖只在 voice worker 绕开了**，未根治。core 的 GEMM 走预编译
   kernel 无碍；将来若触发新的 JIT kernel 会以同样方式失败。把 libc++ 头平铺进 clang
   资源目录**无效**（HIPRTC 搜索路径不含资源目录）。
3. **应用内自更新会冲掉移植**。本包让它恒返回"已是最新"；升级请重装后重跑补丁脚本。
4. **NAR 阶段比官方 CLI 路线慢约 16 s**（占总时长 <5 %），原因未定位。

---

## 许可证与署名

- 上游 t8 代码：见 [`LICENSE`](LICENSE)（Copyright T8star-Aix），本仓库沿用。
- **YuE2 权重：CC BY-NC 4.0 —— 非商用**，见 [`MODEL_LICENSE`](MODEL_LICENSE)。
- SheetSage2 / MERT 见各自 LICENSE；Seed-VC 为 GPL-3.0；Demucs 为 MIT。
- 第三方说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
- 上游项目与作者：<https://space.bilibili.com/385085361>（B站 UP 主 T8star-Aix）。
  这个移植能低成本完成，是因为 t8 的架构（隔离 worker、vendored 依赖、SHA 清单校验）
  本身就把"换一套运行时"变成了一件局部的事。
