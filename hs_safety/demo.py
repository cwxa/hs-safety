"""真实生成的先缓冲、后检测、再释放循环；不把回放标签迁移给新回答。"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from .data import load_tokenizer
from .engine import StreamingDetector
from .extraction import load_backbone
from .features import CausalFeatureBuilder, load_projection
from .io import get_logger, hash_object, save_json
from .models import MODEL_SPECS


class ReleaseBuffer:
    """告警时当前块从未对外发布；释放行为便于独立测试。"""

    def __init__(self):
        self.pending = []
        self.released = []
        self.stopped = False

    def append(self, token):
        if self.stopped:
            raise RuntimeError("Cannot append after a stop decision")
        self.pending.append(int(token))

    def decide(self, alert):
        if self.stopped:
            raise RuntimeError("Stream already stopped")
        if alert:
            discarded = self.pending.copy()
            self.pending.clear()
            self.stopped = True
            return {"released": [], "discarded": discarded}
        released = self.pending.copy()
        self.released.extend(released)
        self.pending.clear()
        return {"released": released, "discarded": []}


def run_demo(config, prompt=None, benchmark=False, monitor=True, save=True):
    import torch

    logger = get_logger("demo")
    options = config.get("demo", {})
    prompt = prompt or options["prompt"]
    name, seed = options.get("model_name", "gru"), int(options.get("seed", 42))
    spec = MODEL_SPECS[name]
    detector = StreamingDetector(config, name, seed) if monitor else None
    builder = CausalFeatureBuilder(load_projection(config), spec.dynamic, spec.final_layer_only) if monitor else None
    tokenizer = load_tokenizer(config)
    model = load_backbone(config)
    device = next(model.parameters()).device
    stride = spec.stride or config["features"]["stride"]
    pids = tokenizer.apply_chat_template(
        [{"role": "system", "content": config["model"]["system_prompt"]}, {"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True)
    if len(pids) > config["data"]["max_prompt_tokens"]:
        raise ValueError("Demo prompt exceeds configured token limit")
    maximum = int(options.get("max_new_tokens", 128))
    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                  else [model.generation_config.eos_token_id])
    events, buffer, response = [], ReleaseBuffer(), []
    monitor_times = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        prefill = model(input_ids=torch.tensor([pids], device=device), use_cache=True,
                        output_hidden_states=False, return_dict=True)
        cache, logits = prefill.past_key_values, prefill.logits[:, -1]
        for step in range(1, maximum + 1):
            # 贪心采样保证同设备下可复现，不通过降安全控制收集生成样本。
            token = int(logits.argmax(dim=-1).item())
            if token in eos_ids:
                break
            response.append(token)
            buffer.append(token)
            output = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                           use_cache=True, output_hidden_states=monitor, return_dict=True)
            cache, logits = output.past_key_values, output.logits[:, -1]
            if monitor:
                hidden = torch.stack([output.hidden_states[layer][0, -1] for layer in config["model"]["layers"]])
            if step % stride == 0 or step == maximum:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                monitor_start = time.perf_counter()
                if monitor:
                    feature = builder.update(hidden.to(torch.float16).cpu().numpy())
                    decision = detector.update(feature)
                    monitor_times.append((time.perf_counter() - monitor_start) * 1000)
                else:
                    decision = {"score": None, "alert": False}
                # 固定长度性能对照只计监测开销；独立guarded演示仍执行真实停止。
                should_stop = decision["alert"] and not benchmark
                released = buffer.decide(should_stop)
                events.append({"position": step, **decision, "released_tokens": len(released["released"]),
                               "discarded_tokens": len(released["discarded"])})
                if should_stop:
                    logger.info("Generation stopped before releasing buffer position=%d score=%.6f threshold=%.6f",
                                step, decision["score"], detector.threshold)
                    break
        # EOS前不足stride的最后一块仍需检测，h_t对应最后已消费的实际回答token。
        if buffer.pending and not buffer.stopped:
            monitor_start = time.perf_counter()
            if monitor:
                feature = builder.update(hidden.to(torch.float16).cpu().numpy())
                decision = detector.update(feature)
                monitor_times.append((time.perf_counter() - monitor_start) * 1000)
            else:
                decision = {"score": None, "alert": False}
            released = buffer.decide(decision["alert"] and not benchmark)
            events.append({"position": len(response), **decision, "released_tokens": len(released["released"]),
                           "discarded_tokens": len(released["discarded"])})
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {"model": config["model"]["id"], "detector": name, "seed": seed, "prompt": prompt,
              "released_text": tokenizer.decode(buffer.released, skip_special_tokens=True),
              "generated_tokens": len(response), "released_tokens": len(buffer.released),
              "token_sequence_sha256": hash_object(response),
              "mode": "baseline" if not monitor else "monitor_only" if benchmark else "guarded",
              "stopped": buffer.stopped, "events": events, "elapsed_seconds": elapsed,
              "generated_tokens_per_second": len(response) / max(elapsed, 1e-9),
              "monitoring_p50_ms": float(np.percentile(monitor_times, 50)) if monitor_times else None,
              "monitoring_p95_ms": float(np.percentile(monitor_times, 95)) if monitor_times else None,
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
              "evidence_scope": "Unlabelled generation; validates integration and buffering, not safety efficacy."}
    if save:
        save_json(Path(config["paths"]["results"]) / "online_demo.json", result)
    logger.info("Released response: %s", result["released_text"])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def benchmark_generation(config, repeats=3, max_new_tokens=64):
    """同一安全说明提示、相同生成长度，对照真实增量生成的监测开销。"""
    from copy import deepcopy

    options = deepcopy(config)
    options["demo"]["max_new_tokens"] = max_new_tokens
    # 预热不计入结果；贪心解码及token散列核验确保对照任务相同。
    run_demo(options, benchmark=True, monitor=False, save=False)
    run_demo(options, benchmark=True, monitor=True, save=False)
    rows = []
    for trial in range(repeats):
        baseline = run_demo(options, benchmark=True, monitor=False, save=False)
        monitored = run_demo(options, benchmark=True, monitor=True, save=False)
        if baseline["token_sequence_sha256"] != monitored["token_sequence_sha256"]:
            raise ValueError("Generation benchmark produced unmatched token sequences")
        rows.append({"trial": trial, "generated_tokens": baseline["generated_tokens"],
                     "token_sequence_sha256": baseline["token_sequence_sha256"],
                     "baseline_seconds": baseline["elapsed_seconds"],
                     "monitor_only_seconds": monitored["elapsed_seconds"],
                     "overhead_ratio": monitored["elapsed_seconds"] / baseline["elapsed_seconds"] - 1,
                     "monitoring_p50_ms": monitored["monitoring_p50_ms"],
                     "monitoring_p95_ms": monitored["monitoring_p95_ms"],
                     "peak_gpu_gib": monitored["peak_gpu_gib"]})
    result = {"trials": rows, "warmup_pairs": 1, "max_new_tokens": max_new_tokens,
              "median_overhead_ratio": float(np.median([row["overhead_ratio"] for row in rows])),
              "note": "Matched-length monitor-only timing; early-stop savings and safety efficacy are not inferred."}
    save_json(Path(config["paths"]["results"]) / "generation_benchmark.json", result)
    return result
