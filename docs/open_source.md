# 开源范围与发布流程

公开仓库为 https://github.com/cwxa/hs-safety ，作者陈纬昕（Weixin Chen），南京邮电大学。原创代码MIT，第三方资源独立适用上游许可。

## 两种交付

- **本地课程归档：**保留学号、处理数据、逐回答分数、检测器权重与内部检查记录，使用 `scripts/package_delivery.py` 打包，仅用于本地研究交付。
- **公开仓库：**通过 `scripts/package_public.py` 从明确的文件白名单导出，保留代码、公开文档、聚合指标、图表和论文；课程学号、内部审稿记录、源数据、逐回答输出、权重、缓存、日志及原工作区Git历史不进入公开树。

公开论文与本地课程稿共享正文、图表和实测结果。公开导出仅移除学号显示和PDF元数据中的学号，保留作者姓名与单位。公开PDF在导出的源码目录重新编译，原课程PDF不被覆盖。

## 验证与发布

```powershell
python scripts/package_public.py --output deliverables/open_source/hs-safety
./scripts/build_papers.ps1 -SkipReport -Tectonic /absolute/path/to/tectonic.exe
python scripts/package_public.py --output deliverables/open_source/hs-safety --finalize
```

第二条构建命令应从导出的项目目录运行；编译工具和缓存放在公开树之外。`--finalize` 检查公开文件范围、学号、明显凭据和个人绝对路径，生成逐文件SHA-256清单；可用 `--verify` 重新核对清单。输出目录已存在时普通导出拒绝覆盖。

发布前在公开树中执行离线测试、Ruff、软件包构建、双语论文检查与相对链接校验。GitHub Actions在Linux和Windows上执行离线测试、Ruff及wheel构建；是否通过以实际远程运行状态为准。

首次公开使用独立Git历史，稳定代码位于 `main`，`develop` 用于集成；贡献经Issue和PR进入，使用Conventional Commits和SemVer。后续维护按相同的公开导出范围审查变更，不能直接推送课程工作区的全部历史。正式tag与GitHub Release引用同一公开提交，附公开源码包和对应校验值。

## 复现与数据

公开发布不附上游数据或模型权重。执行下载脚本获取配置中固定的revision，重建分组样本、PCA、检测器及预测轨迹后，再运行完整报告命令。仓库自带的生成表格与图片支持直接编译论文；重新生成涉及逐回答分析的图表需要先重建预测和特征缓存。

## 分支规则

当前单维护者仓库要求所有变更通过PR进入main，并强制Linux和Windows的离线检查；维护者核对差异、测试和公开范围后合并。禁止强制推送与删除main。审批人数设为0，避免要求同一维护者审批自己的PR；多人维护时可提高独立审批人数。
