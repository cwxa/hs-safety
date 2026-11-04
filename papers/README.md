# 双语论文与实测图表

英文入口 `paper_en.tex` 为 IEEEtran conference 双栏主稿；`paper_zh.tex` 是逐节简体中文对照稿，共享 `refs.bib` 和同一组实验数值。

本次实际编译链为 **Tectonic 0.17.0（XeTeX + 自动 BibTeX）**。源码也支持常规 XeLaTeX/BibTeX，例如 TeX Live 或 MiKTeX。

```powershell
./scripts/build_papers.ps1 -Tectonic /absolute/path/to/tectonic.exe
./scripts/build_papers.ps1 -SkipReport -Tectonic /absolute/path/to/tectonic.exe
./scripts/build_papers.ps1
```

构建前默认从 `configs/main.yaml` 调用 `hs_safety.reporting.generate_report(config)`。`-SkipReport` 仅用于复编排版，避免覆盖手动审阅中的生成文件。正式更新数值时应重新生成报告。

`tables/`、`figures/`、`generated/`、`source_data/` 与 `figure_contracts/` 由真实实验工件生成。缺失测量明确标记 pending，不以模拟数据补齐。探索性终点 PCA/KMeans 模型独立保存在论文源数据目录，仅读训练集拟合，不影响主模型或阈值。

作者信息统一由 `author_info.tex` 提供。公开版本保留姓名与单位，学号仅用于本地课程版。提交前核对预期种子全部完成，并复核最终PDF的引文、图表、字体及术语一致性。
