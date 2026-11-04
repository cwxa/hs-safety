# 数据协议

## 1. 监督语义与来源

`label = 1` 表示完整问答记录的回答不安全，`label = 0` 表示安全。该标签预测目标始终是**该条完整回答的最终风险**，不是每个前缀已有害的人工标注。

主数据使用 BeaverTails 的指定 30k 系列；不混合 30k 与 330k 系列以扩大样本。外测使用 [StreamGuardBench](https://huggingface.co/datasets/Alibaba-AAIG/StreamGuardBench) 中指定的英文 `wildguard/qwen3_8b/test` 数据，仅用于测试。数据来源、revision、上游标签字段、映射和许可须保存在准备产物中；数据版本发生变化时重新执行审计。

BeaverTails 来源：[PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)，许可为 CC BY-NC 4.0。代码 MIT 许可不改变上游数据许可。

StreamGuardBench 固定版本的数据卡声明 Apache-2.0。上游数据卡随本机资源与交付包保留。`data/processed/source_annotations.jsonl` 通过记录 ID 保存原始标签字段、原始值和完整类别字典，独立于检测器输入；对应 manifest 校验其内容及准备审计身份。`prepare` 自动导出，也可运行 `python scripts/export_source_annotations.py` 对现有固定数据补建，期间不修改训练记录。

## 2. 规范记录

| 字段 | 类型/语义 |
| --- | --- |
| `id` | 稳定记录标识；不得依赖随机处理顺序 |
| `group` | 规范化 prompt 的分组标识，用于划分隔离 |
| `split` | `train`、`dev`、`calibration`、`test` 或 `external` |
| `source` | 上游数据来源标识 |
| `source_split` | 上游原始划分，如 `30k_train`；与项目 `split` 分开保存 |
| `label` | 二元最终回答风险；1 为 unsafe，0 为 safe |
| `categories` | 上游风险类别，保留多标签语义 |
| `prompt` | 原始提问文本 |
| `response` | 完整回答文本 |
| `prompt_ids` | 固定 tokenizer 与 chat 模板生成的 prompt token IDs |
| `response_ids` | 不额外加入特殊 token 的回答 IDs |
| `response_len` | `len(response_ids)`；不含 prompt、padding 或 EOS |

字段不齐、标签缺失、空回答、分词结果为空或长度越界时，应在审计中记录排除原因；不得静默替换标签、补造文本或把缺失标签视作安全。

## 3. 划分与长度过滤

- 默认目标数量为 train 8,000、dev 1,000、calibration 1,000；test 保留官方 30k_test 中全部合格记录。实际数量若受过滤或 group 完整性影响，应输出实际值及原因。
- 先建立 prompt group，重复问答去重；train、dev、calibration、test 的 group 两两隔离。测试侧出现的 group 不进入训练侧。外测也报告与主数据的 group 重叠情况。
- 使用相同 tokenizer 过滤到 prompt 不超过 256 token、回答不超过 512 token。不截掉回答尾部后沿用原完整回答标签。
- 对每个划分输出原始数量、去重数量、无效记录、长度排除、group 排除、最终标签比例、回答长度分布及类别计数。
- 标准化、PCA、文本词表或任何数据驱动的预处理只在 train 拟合。dev 仅用于模型选择，calibration 仅用于最终阈值校准。

划分清理中的字符 TF-IDF 近邻审计只读取 prompt 文本，用于避免训练侧与保留测试侧的近重复；其词表不提供给分类模型，也不使用测试标签选择相似度。重复回答模板清理可能改变类别比例，因此论文采用筛选后的实际分布与保留率，不能把结果集称为未经筛选的原始自然分布。

## 4. 检查点与特征契约

检查点位置 `t` 为已观察的回答 token 数，满足 `1 <= t <= response_len`。一个间隔为 `s` 的回答序列采用：

```text
positions = sorted({t : t % s == 0, 1 <= t <= response_len} ∪ {response_len})
```

末尾不足间隔也检查一次；整除时不重复末尾。特征和位置数量必须完全相等；padding 不构成检查点。

原始缓存只保存所选层的因果状态和对齐信息。每层 IncrementalPCA 只使用 train 的检查点拟合，每条训练回答最多贡献 8 个等距检查点。其他划分只执行 transform。PCA 输出按训练集解释方差平方根缩放，即固定 whitening；分母设置数值下限，不能使用当前回答统计量重新归一化。派生 stride 后，按相邻选中状态重新计算差分及漂移，不能把另一间隔的差分直接抽样复用。

设每层降维状态为 `z[t,l,:]`，动态特征按 `[flatten(z), flatten(delta), cosine_drift, norm_difference]` 拼接。后两部分均按层排列：`delta = z[t] - z[t-1]`，`cosine_drift = 1 - dot(z[t], z[t-1]) / max(norm(z[t])*norm(z[t-1]), 1e-8)`，`norm_difference = norm(z[t]) - norm(z[t-1])`。第一个检查点的三种变化量均置零。四层、每层 64 维时静态维度为 256，动态维度为 520。

## 5. 完整性与缓存身份

manifest 需要让运行者判断“这些特征来自哪一份数据、哪个模型、哪一种处理”：schema version、配置摘要、数据标识与划分、模型和 tokenizer revision、模板身份、层索引、基础 stride、dtype、投影身份、记录数、分片清单及完成状态。

已完成分片恢复时应验证文件存在、记录/offset 数一致、最后 offset 与状态数一致、所有位置合法。身份不匹配、半写入分片、重复记录或缺失记录必须显式失败或重新计算，不可静默继续。

原始数据、token IDs、隐状态及模型权重默认仅保存在本地缓存，Git 提交只包含代码、配置、审计摘要和允许发布的聚合结果。日志用记录 ID、计数、参数与耗时定位问题；不要默认打印完整敏感样本。
