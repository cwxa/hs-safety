"""在固定短回答诊断子集上比较真实 KV 回放与已归档离线风险分数。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch

    from hs_safety.config import load_config
    from hs_safety.data import checkpoint_positions
    from hs_safety.engine import StreamingDetector
    from hs_safety.extraction import load_backbone
    from hs_safety.features import CausalFeatureBuilder, load_projection
    from hs_safety.io import configure_logging, get_logger, hash_file, read_jsonl, save_json

    config = load_config("configs/main.yaml")
    configure_logging(config["paths"]["logs"], "streaming_score_alignment")
    logger = get_logger("score_alignment")
    records = read_jsonl(Path(config["paths"]["processed"]) / "test.jsonl")
    chosen = []
    for label in (0, 1):
        candidates = sorted((r for r in records if r["label"] == label and 16 <= r["response_len"] <= 128),
                            key=lambda row: row["id"])
        chosen.extend(candidates[:3])
    if len(chosen) != 6:
        raise ValueError("Six predeclared diagnostic responses are required")
    prediction_path = Path(config["paths"]["results"]) / "predictions/gru/seed_42/test.jsonl"
    offline = {r["id"]: r for r in read_jsonl(prediction_path)}
    detector = StreamingDetector(config, "gru", 42)
    builder = CausalFeatureBuilder(load_projection(config), dynamic=True, final_layer_only=False)
    model = load_backbone(config)
    device = next(model.parameters()).device
    samples = []
    with torch.inference_mode():
        for record in chosen:
            detector.reset()
            builder.reset()
            prefill = model.model(input_ids=torch.tensor([record["prompt_ids"]], device=device), use_cache=True,
                                  return_dict=True)
            cache = prefill.past_key_values
            positions = checkpoint_positions(record["response_len"], config["features"]["stride"])
            scores = []
            for position, token in enumerate(record["response_ids"], 1):
                output = model.model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                                     use_cache=True, output_hidden_states=True, return_dict=True)
                cache = output.past_key_values
                if position in positions:
                    hidden = torch.stack([output.hidden_states[layer][0, -1] for layer in config["model"]["layers"]])
                    decision = detector.update(builder.update(hidden.to(torch.float16).cpu().numpy()))
                    scores.append(decision["score"])
            expected = offline[record["id"]]
            if positions.tolist() != expected["positions"]:
                raise ValueError("KV and offline checkpoint positions disagree")
            errors = np.abs(np.asarray(scores) - expected["scores"])
            mismatches = int(np.sum((np.asarray(scores) > detector.threshold) !=
                                    (np.asarray(expected["scores"]) > detector.threshold)))
            sample = {"id": record["id"], "label": record["label"], "tokens": record["response_len"],
                      "checkpoints": len(scores), "max_absolute_score_difference": float(errors.max()),
                      "mean_absolute_score_difference": float(errors.mean()), "decision_mismatches": mismatches,
                      "positions": positions.tolist(), "kv_scores": scores, "offline_scores": expected["scores"]}
            samples.append(sample)
            logger.info("KV risk diagnostic id=%s checkpoints=%d max_abs_difference=%.6g mismatches=%d",
                        record["id"], len(scores), errors.max(), mismatches)
    save_json(Path(config["paths"]["results"]) / "streaming_score_alignment.json",
              {"samples": samples, "threshold": detector.threshold, "prediction_sha256": hash_file(prediction_path),
               "selection": "First three lexicographically sorted test IDs per label, lengths 16..128; seed 42 GRU.",
               "scope": "Six-response numerical diagnostic, not a population guarantee; frozen trained threshold.",
               "maximum_absolute_score_difference": max(r["max_absolute_score_difference"] for r in samples),
               "total_decision_mismatches": sum(r["decision_mismatches"] for r in samples)})


if __name__ == "__main__":
    main()
