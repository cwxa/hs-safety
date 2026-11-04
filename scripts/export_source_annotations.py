"""从已固定的上游文件导出原标签旁路清单，不改动训练样本或缓存。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from hs_safety.config import load_config
    from hs_safety.data import export_source_annotations
    from hs_safety.io import configure_logging

    config = load_config("configs/main.yaml")
    configure_logging(config["paths"]["logs"], "source_annotations")
    export_source_annotations(config)
