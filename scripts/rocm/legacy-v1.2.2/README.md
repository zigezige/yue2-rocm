# legacy: t8 v1.2.2 三运行时时期的原始脚本

这些是移植**当时实际使用并实测通过**的脚本，原样保留作为记录。

它们假设 t8 v1.2.2 的运行时布局 —— `runtime/core`、`runtime/transcribe`、
`runtime/voice` 三套独立嵌入式解释器。上游此后把三套合并成单个
`runtime/python.exe`（并且换了模型分发方式），所以**这些脚本不要直接用在当前
upstream main 上**；当前布局请用上一级目录的脚本：

| 当前脚本 | 作用 |
|---|---|
| `../setup_rocm_runtime.ps1` | 建单运行时 ROCm 环境（对应上游 `setup_unified.ps1`） |
| `../apply_rocm_port.py` | 幂等应用全部移植补丁 |
| `../patch_audiotools.py` | 站点包补丁（无法进仓库的那一处） |
| `../fetch_mirror_models.py` | 镜像分块续传下载模型 |
| `../verify_capabilities.py` | doctor + 转谱渲染 + 音色转换端到端验证 |

三运行时时期的实测数据见 `../../docs/PORT_REPORT.md`。
