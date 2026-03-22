# AGENTS.md

Instrumentation discipline for this repository:

1. Preserve limit-focused logs in `train_smoke.py` and `metrics_logging.py`.
2. Any training-loop change must keep (or improve) visibility for:
   - per-optimizer-step timing
   - memory snapshots at key lifecycle checkpoints
   - sequence length statistics (including p95)
   - effective batch / gradient accumulation metadata
   - checkpoint size + save duration
   - eval timing split (reload, eval, generation when applicable)
   - preprocessing and dataloader overhead
3. Keep logs human-readable for GitHub Actions and keep the machine-readable artifact compact JSON under `outputs/metrics/`.
4. Avoid logging full sample text, huge configs, or per-token dumps.
5. If new scaling dimensions are introduced (e.g., longer context, larger model, multi-checkpoint strategy), extend both logs and JSON schema so bottleneck diagnosis remains straightforward.
6. Prefer additive instrumentation with minimal runtime overhead; do not change model behavior unless required for measurement correctness.
