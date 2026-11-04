# 基于隐状态流式序列的大模型内容安全数据挖掘与检测研究

[English](README.md) · [复现指南](docs/reproducibility.md) · [课程方法对应](docs/数据挖掘方法与课程对应.md) · [中文论文](papers/paper_zh.pdf) · [英文论文](papers/paper_en.pdf)

本项目以大模型内容安全为应用场景，将冻结Qwen2.5的因果隐状态视为高维序列数据，完成从数据准备、PCA降维、特征工程、分类检测、聚类分析到统计评价的完整数据挖掘流程。监督目标为**根据已经观察到的回答前缀，预测完整回答的最终安全标签**。

## 项目成果

- **数据方法完整：**分组清洗与去重、增量PCA、动态特征、TF–IDF、逻辑回归、MLP、GRU以及K均值分析。
- **实验可追溯：**13,610条合格回答，两种骨干规模，三个固定种子，共36个检测器实例；保留主对照、消融、独立校准和外部评估。
- **工程闭环：**分片缓存、签名校验、断点恢复、共享离线／在线特征，以及缓冲检查释放接口。
- **论文与代码共享结果源：**聚合指标、图表源数据及中英文IEEEtran双栏稿一并提供。

共同5%校准目标下，0.5B静态探针的Recall@128为43.7%，实际测试FPR为5.60%；GRU分别为38.1%和3.27%。1.5B静态探针达到54.0%召回和5.75% FPR。完整结果见[实测摘要](results/summary.md)和[指标文件](results/metrics.json)。这些运行特征为按召回、误报与资源需求选择检测器提供依据。

## 快速运行

推荐Python 3.12。建立并激活虚拟环境后，按英文首页的CPU安装命令运行；默认测试不下载数据或模型，使用合成输入和微型随机模型。GPU实验应先安装适配驱动的CUDA版PyTorch。

```sh
python -m hs_safety --help
python -m pytest -m "not integration" --strict-markers -q
python -m ruff check hs_safety scripts tests
```

正式实验按[复现指南](docs/reproducibility.md)下载固定版本资源并执行 `prepare → validate → extract → train → calibrate → evaluate → demo → report`。主实验在6 GB显存的RTX 3060 Laptop上完成。可选1.5B实验复用主数据划分。

## 公开范围

公开代码、配置、聚合结果、图表与论文。原始／处理后的样本、逐回答预测、骨干与检测器权重、隐状态缓存、本地日志及课程归档由下载和实验步骤在本机生成。开源论文保留姓名与单位，学号仅留在本地课程版本；公开仓库使用独立导出历史。

作者：**陈纬昕，南京邮电大学**。引用信息见[CITATION.cff](CITATION.cff)。原创代码使用[MIT许可](LICENSE)，外部资源独立适用其上游条款，详见[第三方声明](THIRD_PARTY_NOTICES.md)。协作、安全及发布方式见[贡献指南](CONTRIBUTING.md)和[开源说明](docs/open_source.md)。
