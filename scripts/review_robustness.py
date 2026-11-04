"""复用冻结预测结果执行审稿补充分析，不训练或重写主实验。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from hs_safety.config import load_config
    from hs_safety.io import configure_logging
    from hs_safety.review_analysis import run_review_analysis

    config = load_config("configs/main.yaml")
    configure_logging(config["paths"]["logs"], "review_robustness")
    run_review_analysis(config)
