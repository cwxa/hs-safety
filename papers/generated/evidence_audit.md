# Evidence audit

Measured in-domain results are available for 3 of 3 configured dynamic-GRU seeds.

- Tables: results/metrics.json, per-seed split measurements.
- Figures: numeric metrics or deidentified prediction trajectories; architecture is a schematic.
- Sample counts: normalized processed JSONL files; no raw response text is exported.
- Inference target: eventual response safety, not token or onset labels.
- Calibration target: empirical sequence-level FPR on the calibration sample, not a test guarantee.
- Missing and undefined metrics remain missing; no synthetic measurements are inserted.
- Author metadata are supplied by the author and shared through papers/author_info.tex.
