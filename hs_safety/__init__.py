"""基于因果隐状态序列的最终回答风险预测。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hs-safety")
except PackageNotFoundError:
    __version__ = "0+uninstalled"
