"""统一研究命令入口；各阶段可独立重跑和审计。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import ensure_directories, load_config
from .io import configure_logging, get_logger, read_jsonl, timed


def main():
    parser = argparse.ArgumentParser(description="Causal hidden-state final-response risk experiments")
    parser.add_argument("command", choices=["prepare", "extract", "train", "calibrate", "evaluate", "demo", "benchmark", "report", "run", "validate"])
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--prompt")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    configure_logging(config["paths"]["logs"], args.command)
    logger = get_logger("cli")
    with timed(logger, args.command, config=args.config):
        if args.command in ("prepare", "run"):
            from .data import prepare_data
            prepare_data(config)
        if args.command == "validate":
            from .extraction import validate_causal_alignment
            validate_causal_alignment(config, read_jsonl(Path(config["paths"]["processed"]) / "train.jsonl"))
        if args.command in ("extract", "run"):
            from .extraction import extract_features
            from .features import build_features
            extract_features(config)
            build_features(config)
        if args.command in ("train", "run"):
            from .engine import train_all
            train_all(config, models=args.models, seeds=args.seeds)
        if args.command in ("calibrate", "run"):
            from .engine import calibrate_all
            calibrate_all(config)
        if args.command in ("evaluate", "run"):
            from .engine import evaluate_all
            evaluate_all(config)
        if args.command == "demo":
            from .demo import run_demo
            run_demo(config, prompt=args.prompt)
        if args.command == "benchmark":
            from .demo import benchmark_generation
            benchmark_generation(config)
        if args.command in ("report", "run"):
            from .reporting import generate_report
            generate_report(config)


if __name__ == "__main__":
    main()
