# YuE2 on AMD RX 9070 XT — 部署记录

实测环境：Windows 11 (build 26200) / AMD Radeon RX 9070 XT 16GB (gfx1201) /
原生 Windows ROCm，**无 CUDA、无 Triton、无 flash-attn**。

## 结论

YuE2 在这张卡上**可以正常出歌**。官方 README 写的 "Linux + NVIDIA + 24GB VRAM"
是推荐起点，不是硬门槛——24GB 那个数字在代码里只是 `memory_budget_gib` 参数的
默认值，而且 `pipeline.py` 会自动按实卡显存裁剪：

```python
budget = min((self.memory_budget_gib - 2) * 2**30, total - 2 * 2**30)
```

16GB 卡上预算被夹到 13.92 GiB，实测够用。

## 三个必须知道的坑

### 1. CUDA graph 在 ROCm 上会误判后端并崩溃（已打补丁）

`src/yue2/cuda_graph.py` 用 `device.type == "cuda"` 判断，而 ROCm 下这个值就是真。
它只检查 ATen **schema** 里有没有 `seqused_k` 就选 flash 后端，**不看后端实现能力**，
而 schema 是跨后端共享的。实测：

```
RuntimeError: [ROCm] mha_varlen_fwd: seqused_k must be nullopt
```

旁注：这个 ROCm 版本里 `torch.backends.cudnn.is_available()` 返回 **True**，但
`CUDNN_ATTENTION` 实际是 RuntimeError，所以 cudnn 回退也指望不上。

**补丁**（本目录 `YuE/src/yue2/cuda_graph.py` 已应用）：`attention_backend == "auto"`
时在 HIP 上强制走掩码 SDPA。**不能**改成给 `seqused_k=None` 绕过去——那样变长 FA
会去注意未使用的未来 cache 槽位，算出错误结果。

已验证掩码 SDPA 在 CUDA graph 内：捕获成功、replay 响应原地更新的 positions、
与 eager **数值差 0.000e+00**。

### 2. 模型必须走 hf-mirror 直连，且不能用 huggingface_hub

| 目标 | 结果 |
|---|---|
| `huggingface.co` 取 6.76GB 权重 | ❌ `RemoteDisconnected`（小 API 请求能通，大文件被重置） |
| `hf-mirror.com` 取权重 | ✅ HTTP 206，单流 ~5 MiB/s，6 路并发峰值 60 MiB/s |

但 `huggingface_hub` 对镜像不兼容（元数据层 `LocalEntryNotFoundError`），
且 Windows 无开发者模式下缓存建符号链接会 `PermissionError`。
**解决**：绕开 hub，用标准 HTTP 分块续传（见部署脚本 `mirror_download_v2.py`）。
最终 6.76GB 主权重 9.2 分钟下完。

### 3. 不能装进 ComfyUI 便携包

- ComfyUI 里是 `transformers 5.3.0`，YuE2 锁 `4.57.6`；且 YuE2 用
  `torch_dtype=`（v5 已改名 `dtype=`）等旧 API。
- 社区那个 ComfyUI 节点（`smthemex/ComfyUI_YuE`）包的是 **YuE v1 不是 YuE2**，
  而且硬编码 `attn_implementation="flash_attention_2"`（无回退）、默认开
  mmgp 和 `torch.compile(max-autotune)`，int8/exllamav2 量化在 ROCm 上全不可用。

## 安装要点（复现用）

```powershell
# 1. venv（官方推荐 3.12）
& 'D:\anaconda3\python.exe' -m venv D:\YuE2\venv
$py = 'D:\YuE2\venv\Scripts\python.exe'
& $py -m pip install --upgrade pip uv

# 2. AMD TheRock ROCm（不是 PyPI 的 CUDA 版！）
& $py -m uv pip install --extra-index-url https://rocm.nightlies.amd.com/v4/whl/ --pre "rocm[libraries,device-gfx1201]"
& $py -m uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch

# 3. YuE2 依赖（绝不能让它碰 torch）
& $py -m uv pip install transformers==4.57.6 huggingface-hub==0.36.2 safetensors==0.7.0 `
    tiktoken==0.12.0 "numpy==2.2.6" soundfile==0.13.1 accelerate==1.13.0

# 4. 关键：--no-deps，否则 pip 会用 PyPI 的 torch==2.10.0 覆盖掉 ROCm torch
& $py -m uv pip install --no-deps --editable D:\YuE2\YuE
```

装完验证：

```
torch 2.11.0+rocm7.13.0a20260416   hip 7.2.0
transformers 4.57.6                numpy 2.2.6
available True                     bf16 True
device AMD Radeon RX 9070 XT       yue2 import OK
```

## 官方前端情况

官方仓库**没有本地 WebUI**。提供的是：
- CLI：`yue2 generate --request examples/song.json --output outputs/song`
- Python 分阶段 API：`plan() → generate_semantic() → synthesize() → decode()`
- 官方 agent skill：`skills/yue2-music/SKILL.md`（可被支持 SKILL.md 的 agent 直接使用）
- 在线 demo：https://map-yue2.github.io/（不是本地）

社区有 `T8mars/Comfyui-YuE2-T8`，带独立本地 WebUI（`127.0.0.1:8189`，不需要
ComfyUI，靠隔离 worker 避免污染 ComfyUI 环境）。但它的安装脚本把三套内嵌
Python 全锁在 `--index-url .../whl/cu128`，AMD 上**没有安装路径**，需要按上面的
ROCm 源重写；且该项目零 AMD 支持、零 AMD 社区反馈，仅在 RTX 5090 Laptop 24GB
上验证过。它的 `torch.cuda.is_available()` / BF16 断言在 ROCm 上恰好能过，
默认 `backend="torch-eager"` 也绕开了 CUDA graph 坑——真正的门槛只有 cu128 轮子。

## 性能实测（16GB / ROCm）

同一首《city_lights》示例（示例歌词自然结束，均未截断），两种后端全长度对比：

| 阶段 | `torch`（CUDA graph） | `torch-eager` |
|---|---:|---:|
| 歌曲时长 | 63.68s | 59.96s |
| ABC 乐谱规划 | 28.5s（18.5 tok/s，528 tok） | 24.4s（22.1 tok/s，540 tok） |
| 语义 token 生成 | 111.3s（**14.3 tok/s**，1593 tok） | 60.7s（**24.7 tok/s**，1500 tok） |
| NAR flow matching | 6.36s（32 步） | 5.94s（32 步） |
| VAE 音频解码 | 165.6s / 4 chunk = 41.4s | 165.8s / 3 chunk = 55.3s |
| **e2e 总计** | 321.3s | **267.9s** |
| 进程退出码 | 1 | **0** |

折算：graph 5.05 秒/音频秒，eager 4.47 秒/音频秒 —— **eager 快约 12%**，
且能干净退出。所以本机默认用 `--backend torch-eager`。

### 为什么 CUDA graph 在长序列上反而更慢

`GraphAR` 的 `capacity = max(len(prefix)) + max_tokens`，而掩码 SDPA 分支
**每一步都对整个 capacity 算注意力**。上游 flash 路径之所以没这个问题，是靠
`seqused_k` 告诉 FA 真实有效长度——**而 ROCm 拒绝的正是这个参数**。
所以：

- 短序列（max_tokens=600，capacity≈1030）：graph 46.3 tok/s vs eager 25.0 → graph 快 1.85×
- 长序列（max_tokens=9000，capacity≈9669）：graph 14.3 tok/s vs eager 24.7 → graph 慢 42%

结论：**graph 只在小 capacity 时划算**；生产参数下 eager 更优。

### 一个容易误判的坑：第一次 VAE 解码会特别慢

首次运行 VAE 解码耗时 139.3s（2 chunk，69.7s/chunk），之后同样工作是
41–55s/chunk。这不是后端差异，而是 **MIOpen 首次为卷积 solver 付的一次性
调优开销**（之后落盘缓存）。排坑时不要把首跑的慢当成配置问题。

日志里反复出现的 `MIOpen(HIP): Warning [IsEnoughWorkspace] Solver <GemmFwdRest>,
workspace required: 1816543232, provided ptr: 0x0000000000000000 size: 0` 是
solver 选型/workspace 相关的良性告警，实测不影响结果，可用 `MIOPEN_FIND_MODE`
进一步调优。

### 退出码 1 说明

`--backend torch` 时进程在**所有产物已写出并完成哈希校验之后**、解释器拆卸阶段
崩溃，返回 1。`result.json` 的 `status` 仍为 `complete`、`audio.flac` 完整可播，
所以是纯收尾问题，不是推理失败。`torch-eager` 无此问题。
控制实验确认基础 torch / bf16 matmul / SDPA / CUDA graph 单独跑都 `exit 0`，
只有该路径组合会触发。

## 语言支持：单权重双语（中文无需额外模型）

YuE2-3B 的模型卡 frontmatter 就是 `language: [zh, en]`——**一个权重同时支持中英文**。

⚠️ **不要照搬 YuE v1 的经验**：v1 确实分了 en / zh / jp-kr 专用权重
（`YuE-s1-7B-anneal-zh-cot` 等），但 **YuE2 只发布了 `m-a-p/YuE2-3B` 一个权重，
没有语言分支**。本机已下载的这份就是全量的，不需要再下任何中文模型。

语言通过 `style` 字段的标签声明，没有单独的语言参数：

```json
{"style": "Mandarin, warm piano, acoustic pop, female vocal, 84 BPM", "lyrics": "..."}
```

旁证：
- 官方 CLI 的内置默认请求本身就是中文歌（`src/yue2/cli.py:108`，
  style 为 `"Mandarin, warm piano, acoustic pop, female vocal"`）。
- 分词器对 CJK 有独立字表：实测中文 7 字 → 6 token（≈1 token/字），
  token ID 落在 99xxx–119xxx 区间；日文韩文同样有独立区间（12xxxx）。
  而英文 25 字符才 6 token。
- 模型卡 demo 含 `Mandarin funk / nu-disco` 条目，官方 agentic demo 也是
  "from Mandarin pop to English jazz"。

## 优化空间（实测）

### VAE 解码分块大小：512 已是最优档

同一段 1499 帧（60s 音频）latent，只跑 decode：

| tiles | core_frames | 解码耗时 | 峰值显存 | RMS |
|---:|---:|---:|---:|---:|
| 12 | 128 | 104.0s | 1.27 GB | 0.0942 |
| 6 | 256 | 161.6s | 1.88 GB | 0.0942 |
| **3** | **512** | **101.6s** | 3.11 GB | 0.0942 |
| 2 | 1024 | 230.9s | 5.50 GB | 0.0942 |
| 1 | full（不分块） | 285.0s | 7.70 GB | 0.0942 |

五者 RMS 完全一致，说明只是速度差异、不影响音质。

**注意上游默认值对这张卡是慢的**：`vae_core_frames` 在
`memory_budget_gib > 12` 时取 **1024**（为 24 GB 卡选的），在本机比 512 慢 2.3×。
本目录的 `yue2_run.py` 已固定 512。

非单调（256 比 128 和 512 都差）说明主导因素是 **MIOpen 按张量形状挑 solver**，
而不是干净的 O(T²) 关系。所以换块大小是在赌 solver 选择，512 已在好档位，不必再调。
如需进一步压榨，可试 `MIOPEN_FIND_MODE=3` + `MIOPEN_FIND_ENFORCE=3`
（ComfyUI 里的"調優模式"，首跑慢、之后走缓存）。

### FP8 量化：门槛会通过，但结果是错的，绝对不要开

`quantization.py` 的门槛是 `torch.cuda.get_device_capability(device) >= (8, 9)`，
而本机报 **(12, 0)** ⇒ **门槛通过**，`torch._scaled_mm` 也存在，看起来一切正常。

但实测（四种形状，含真实的 2048 hidden / 184704 vocab）：

| 形状 | vs 反量化参考 |
|---|---:|
| M=16 K=2048 N=512 | 99.99863 |
| M=16 K=2048 N=2048 | 109.82619 |
| M=1 K=2048 N=184704（lm_head） | 116.88484 |
| M=1500 K=2048 N=2048（prefill） | 89.68730 |

对比基准是**反量化后的权重**，也就是把 FP8 量化误差排除掉、只检验 `_scaled_mm`
算得对不对。相对误差约 90–117 意味着输出量级完全不对——**不是精度损失，是计算错误**。
所以在 ROCm 上开 `quantization="fp8"` 会**静默产出垃圾音频**。

（首次探测曾误报，原因是我把权重写成了 `wt.t().contiguous().t()`——双重转置等于
`wt`，于是拿 `a @ w` 去对 `a @ w.t()`，误差自然荒谬。修正后才确认上面这个结论。）

### 环境细节：MIOpen 需要写权限

MIOpen 会把 conv solver 调优结果写进 SQLite 库。如果进程没有写该目录的权限，会直接失败：

```
MIOpen Error: sqlite_db.cpp:224: Internal error while accessing SQLite database:
unable to open database file
RuntimeError: miopenStatusInternalError
```

正常运行（普通用户权限跑 `run_yue2.bat`）不会遇到；但任何受限沙箱/只读环境下跑
YuE2 都会撞上，特征就是 conv1d 抛 `miopenStatusInternalError`。



## 出歌速度实测（Patch 2 前后对照）

同一请求（`zh_song.json`，seed 20260917，产出 174.919 s 音频）：

| 运行 | wall | VAE 解码 | vae_frames |
|---|---:|---:|---|
| T8 Patch 2 之前 | 613.7 s | 313.0 s | 1024 |
| T8 Patch 2 之后 | **393.4 s** | **107.0 s** | **512** |
| 官方 CLI 直跑 | 423.4 s | 155.8 s | 512 |

**−220.3 s（−35.9%）**，差距几乎全在 VAE 分块 → 诊断成立。折算 3.51 → 2.25 s/音频秒。

⚠️ **跨天比较有系统性漂移**：同是 512 分块，早期 VAE 155.8 s、本次 107.0 s
（最可能是 MIOpen solver 调优缓存落盘）。所以同日同会话的 A/B 才可靠，
与历史数字比较留 ~15% 余量。这也再次印证「首次 VAE 解码异常慢」那条。

## 补丁脚本的两条纪律（血泪）

**1. 必须保留原文件换行符。** 早期版本用 Python `Path.write_text()` 写回，
Windows 上会把 `\n` 翻译成 `\r\n` —— 一个 2 行修复变成整文件 diff（git 里满屏变化）。
实测：`transcribe_worker.py` 从 4836 B 涨到 4957 B（+121 = 15 B 文案 + 106 个多余 CR）。
统一改用 `open(..., newline="")` 读写后，回放产出与实测 kit **逐字节一致**（7 文件 SHA256 相同）。

**2. 同一文件多锚点时，幂等标记要选"全部改完才出现"的特征串。**
在 `vendor/seed-vc/inference.py` 上踩过：先用补丁 A 的注释做标记，后来新增补丁 B 时
脚本认为"已打过"而整体跳过 → B 漏打。换成用 B 的特征串做标记后正常。

## .bat 文件闪退（双击即退）—— ASCII 铁律

**现象**：双击 un_yue2.bat / start_rocm.bat 窗口瞬间关闭或报错刷屏后退出。

**根因**：bat 是 **UTF-8 保存且含中文**。cmd 按当前代码页逐行切分批处理，
chcp 65001 虽在第 2 行生效，但 cmd 的行读取对多字节字符**错位**，会把一行
从中间切成两行 —— REM ...kernels... 被切成 REM + kernels...，后者被当命令执行：
`
'kernels' is not recognized as an internal or external command
'IMENTAL' is not recognized ...        ← EXPERIMENTAL 被切断
'目录>' / '义歌词' is not recognized   ← UTF-8 中文被切坏
`

**铁律：.bat 文件里只放 ASCII。** 中文提示移到 Python 里输出（Python 处理 UTF-8 没问题）。
两个启动器已重写为纯 ASCII（un_yue2.bat、start_rocm.bat），部署前用
(文件字节中 >127 的数量) == 0 校验。

**另一个相关坑**：Windows 会锁定**正在运行的 .bat 文件**——双击后停在 pause 的窗口
会让该文件无法覆盖（Copy-Item: Access denied）。先关掉那个窗口/结束 cmd 进程再部署。
