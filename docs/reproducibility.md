# 复现指南

## 1. 环境准备

使用 Python 3.12，在项目根目录建立独立虚拟环境。GPU 版 PyTorch 应与已有 NVIDIA 驱动兼容；本项目开发环境为 PyTorch 2.7.0 + CUDA 12.6。不要在已有环境中无提示替换 GPU PyTorch 为 CPU 版。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m hs_safety --help
```

锁文件由 `scripts/record_environment.py` 根据实测依赖闭包生成，适用于 Windows/Python 3.12/CUDA 12.6。CPU 环境可按 CI 安装 CPU PyTorch 后安装项目；跨平台不直接沿用 Windows GPU 锁文件。本次运行复用宿主已验证的 CUDA PyTorch，虚拟环境启用了 `system-site-packages`，其余覆盖版本与资源校验记录见 `results/environment_packages.json`。新机器优先使用上述独立环境。数据与模型首次下载需要网络；准备完成后各实验阶段使用本地工件。

## 2. 运行顺序

所有阶段使用 `--config configs/main.yaml` 指定同一份配置。先通过 CLI 帮助确认实际子命令和阶段参数；正式流程按下表执行，不能跳过数据审计或以测试集替代校准集。

```powershell
python scripts/download_assets.py --config configs/main.yaml
python -m hs_safety prepare --config configs/main.yaml
python -m hs_safety validate --config configs/main.yaml
python -m hs_safety extract --config configs/main.yaml
python -m hs_safety train --config configs/main.yaml
python -m hs_safety calibrate --config configs/main.yaml
python -m hs_safety evaluate --config configs/main.yaml
python -m hs_safety demo --config configs/main.yaml
python -m hs_safety benchmark --config configs/main.yaml
python -m hs_safety report --config configs/main.yaml
```

`extract` 一次完成原始检查点提取、PCA 拟合和投影缓存构建。`train` 可用 `--models` 与 `--seeds` 明确限定实验；例如先验证 `--models static_lr gru --seeds 42`，正式结果再运行完整配置。`run` 连续执行准备、提取、训练、校准、评价和报告，不包括资源下载、因果验证与生成演示。

| 顺序 | 阶段 | 继续前必须核对 |
| --- | --- | --- |
| 1 | 环境与资源检查 | Python/PyTorch/驱动、GPU 可用性、磁盘峰值空间 |
| 2 | 数据准备 | 来源 revision、标签映射、去重、分组隔离、长度过滤 |
| 3 | 小样本特征探测 | 层/位置/KV 对齐、显存、耗时、缓存身份 |
| 4 | 正式原始状态提取 | 各划分记录数与分片清单一致，可断点恢复 |
| 5 | train-only PCA 拟合与缓存压缩 | 每条 train 最多 8 个检查点；其他划分仅 transform |
| 6 | 基线与 GRU 训练 | seed、训练配置、dev 选择依据、模型工件 |
| 7 | 阈值校准 | 使用 calibration 的安全回答整段最大分数 |
| 8 | test 与 external 评价 | 模型及阈值固定，保存逐回答预测 |
| 9 | 真实生成演示 | 先缓冲后检查再释放，尾块与停止路径通过 |
| 10 | 图表与论文 | 所有结果来自实际运行产物 |

当实现输出说明与本文存在差异时，应先修正文档或接口并记录版本，避免用同一实验标识混跑不同协议。

## 3. 最小复现清单

每次正式运行至少保存：完整配置及摘要、代码版本（有 Git 时）、Python 与依赖版本、设备信息、随机 seed、数据审计、模型/tokenizer revision、特征/PCA 清单、训练日志、选择的 checkpoint、校准阈值、逐回答预测、聚合指标及计时结果。

随机 seed 能控制抽样、初始化和批顺序，但不同硬件或库版本仍可能产生浮点差异。数值对齐测试使用与 dtype 相适应的容差；不将跨设备逐位一致作为默认要求。

## 4. 测试

```powershell
python -m pytest -m "not integration"
```

默认测试使用合成数据和微型随机模型；离线、无 API key、无外部模型缓存时也应通过。标记为 `integration` 的测试只在已明确准备本地工件和设备后执行，不属于 CI 的网络下载路径。

CI 先验证 Windows/Python 3.12 的 CPU 测试，再由本地 GPU 冒烟运行验证正式骨干、显存和生成循环。单元测试通过不能替代真实数据审计或 GPU 检查。

## 5. 论文编译与结果更新

英文正式论文和逐节中文对照均使用 `IEEEtran` 双栏，参考文献采用 IEEEtran BibTeX。中文版本使用 XeLaTeX 与中文字体支持，保留 IEEEtran 文档类；两稿共享图表和结果表，不手工维护两份互不关联的实验数值。

本机若未安装 TeX，应在实施初期验证便携工具链或其他已授权编译环境，至少完成一次最小中英文编译。最终检查须包含引用解析、公式/图表溢出、双栏跨栏布局、字体和中文字符、图表清晰度以及两稿数值一致性。未完成的实验保持明确标记，不用模拟数字补齐结果。

本次使用便携 Tectonic 0.17.0（集成 XeTeX 与 BibTeX），可运行：

```powershell
.\scripts\build_papers.ps1 -Config configs/main.yaml -Tectonic .tools/tectonic/tectonic.exe
```

脚本先刷新结果工件；提供 `-Tectonic` 时由其自动处理多遍编译，否则使用 `-XeLaTex` 与 `-BibTex` 指定的工具。`-SkipReport` 用于仅排版复查，`-Python` 用于指定实验环境。两份 PDF 均须逐页视觉检查。`benchmark` 使用相同安全提示与贪心生成序列测量无监测和仅监测的固定长度开销；实际停止行为单独记录于 `online_demo.json`。

便携工具可从 [Tectonic 官方发布页](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0)取得 Windows x86_64 版本，解压至 `.tools/tectonic/`。首次编译需要联网取得公开 TeX 包；默认缓存为项目内 `.cache/tectonic`，已设置 `TECTONIC_CACHE_DIR` 时尊重该路径。交付包包含源码与最终 PDF，复编所需工具及 TeX 缓存由本机保留或按上述方式重新准备。

```powershell
python scripts/verify_papers.py
```

该命令核对文本标记与纸面边界，并在 `papers/qa/` 生成所有页面及联系表。自动检查不能替代人工检查图表清晰度、分栏和字符显示。

## 6. 规模验证与 Git 协作

1.5B 复用主实验已处理样本，避免第二次划分。依次运行 `scripts/download_scale.py`、`scripts/precheck_scale.py --run-gpu`；预检通过后，以 `configs/scale.yaml` 运行 `extract/train/calibrate/evaluate`。禁止对规模配置执行 `prepare` 或 `run`，它们共享主数据目录。

工作分支为 `feature/streaming-hidden-state-safety`，`develop` 用于集成，`main` 保留稳定版本。启用 `git config core.hooksPath .githooks` 可阻止直接向 main 提交。提交遵循 Conventional Commits，合并前检查测试与代码审查；远程仓库建立后由托管平台配置 PR 和分支保护。

## 7. 交付包

`python scripts/package_delivery.py` 将代码、配置、中英文论文、所有实测指标与逐回答分数、已处理划分及轻量检测器权重打包至 `deliverables/hs_safety_research.zip`。清单包含逐文件 SHA-256、代码提交号、测试与图表检查摘要，压缩包完成后执行 CRC 检查并保存整包校验值。执行前需要 `results/test_report.xml`、`papers/qa/paper_audit.json` 及最终图表审查记录；可通过 `pytest --junitxml=results/test_report.xml` 和上述 PDF 检查入口生成对应工件。

包内不含多 GB 骨干、隐状态缓存、虚拟环境或 TeX 工具链。上游数据卡与固定 revision 一同保留，数据的独立许可仍然适用。全部文件保存在本机，此打包操作不上传或公开发布内容。

## 8. 公开仓库与本地课程归档

GitHub公开版本使用白名单导出，保留代码、聚合结果和论文，独立于第7节的本地完整归档。公开树中不含逐回答预测、处理数据、权重及内部审稿记录；完整报告生成需要先重建这些工件。公开范围、学号处理和首次发布流程见[开源说明](open_source.md)。
