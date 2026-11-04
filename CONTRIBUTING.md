# 贡献指南

欢迎提交能够复现问题、改进实验可信度或降低资源需求的贡献。提交前先阅读 [README](README.md)、[架构设计](docs/architecture.md) 和[实验协议](docs/experiments.md)。

## 开发流程

1. 先建立 Issue，说明问题、预期行为、复现条件及验收方式；简单文档修正可直接提交 PR。
2. `main` 用于稳定版本，`develop` 用于集成；使用 `feature/`、`bugfix/`、`hotfix/` 或 `release/` 分支。Fork 工作流保持上游同步。
3. 使用 Conventional Commits，保持原子提交，例如 `feat: add causal checkpoint extraction` 或 `fix: preserve final stream checkpoint`。
4. PR 关联 Issue，描述最终行为、必要的设计取舍、验证结果及资源影响。通过 review 和 CI 后合并；hotfix 发布后回合并到集成分支。
5. 发布使用 SemVer、CHANGELOG 和 tag。未经授权不直接推送、合并或发布。

仓库维护者应通过托管平台配置 `main` 分支保护；本地文档不等同于已经启用该设置。

## 代码与测试

- 保留准确的已有注释；新增解释性注释使用中文，打印和运行日志使用英文。
- 日志记录参数、变量、阶段、样本计数、错误原因与关键耗时。使用合适的 info/warning/error 等级，避免默认打印完整敏感文本。
- 模型输入保持因果性；预处理只在 train 拟合；不以测试集选择配置。
- 默认测试不得依赖网络、个人路径、真实模型缓存或 GPU。涉及正式模型的检查使用独立 integration 标记。
- 增加与改动风险相匹配的测试，重点验证数据泄漏、索引、恢复和停止语义；不为简单文档修改堆砌测试。

```powershell
python -m pip install -e ".[dev]"
python -m pytest -m "not integration"
```

## 数据、结果和许可

不提交 API key、个人凭据、原始数据、模型权重或大型隐状态缓存。共享数据时保留来源和上游许可，确认有权再分发。对外发布聚合结果时，记录数据 revision、配置、seed 与实际运行环境。

不得伪造实验结果、隐去失败运行、把最终回答标签冒充 token 级标注，或在未完成测试时宣称已验证。任何变更若改变数据、监督语义或评价口径，应更新协议、配置版本与变更日志。
