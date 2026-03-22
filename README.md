# Tiny Distill Limit Logging Smoke Test

This repository provides a CPU-friendly tiny distillation-style smoke test with verbose instrumentation tailored for GitHub Actions logs.

## Run locally

```bash
python3 scripts/train_smoke.py --samples 64 --batch-size 4 --grad-accum 4
```

Metrics are emitted to stdout and also written as JSON under `outputs/metrics/latest_metrics.json`.

## How to read the limit logs

Use the logs to identify the first scaling bottleneck when increasing sample count (64 -> 128 -> 256):

- **Runtime scaling limit**
  - Check `optimizer_step` lines for `step_duration_s` trend and `train_duration_s` in JSON.
  - If step duration grows superlinearly while memory is stable, runtime is the limiting axis.
- **RAM / memory headroom limit**
  - Check `memory_snapshot` at `after_model_load`, `after_first_forward_backward`, per `optimizer_step`, and eval/save boundaries.
  - If RSS (`rss_mb`) approaches system availability (`available_mb`), memory is the next hard limit.
- **Token-length-driven cost limit**
  - Inspect `sequence_stats` (`input_tokens` and `target_tokens`, especially `p95`) and `tokens_per_optimizer_step`.
  - Rising p95 token lengths usually increase forward/backward CPU time before sample count does.
- **Checkpoint IO limit**
  - Use `checkpoint.save_duration_s`, `checkpoint.size_mb`, and `checkpoint.file_count`.
  - If save duration is a meaningful part of total runtime, checkpointing becomes the next scaling bottleneck.
- **Eval overhead limit**
  - Compare `eval.reload_s`, `eval.eval_duration_s`, and `eval.generation_duration_s` with `train_duration_s`.
  - If eval share of wall time grows as dataset size increases, reduce eval frequency or payload.
- **Data pipeline overhead**
  - Track `tokenization_preprocessing_s` and `dataloader_init_s`.
  - If these dominate early, optimize preprocessing before increasing training scale.

### Why 64 is safe, 128 plausible, 256 stretched

This instrumentation is designed so one CI run can answer that question directly:

1. **64 samples safe** if memory snapshots are flat, checkpoint/eval overhead are small, and optimizer step durations are stable.
2. **128 plausible** when runtime roughly doubles with no abrupt p95 token or memory jump.
3. **256 stretched** if any of these inflect: per-step duration, RSS pressure, save/eval share, or token-length p95.

## JSON artifact shape (compact)

`outputs/metrics/latest_metrics.json` contains:

- `events`: timestamped human-readable timeline entries
- `memory_snapshots`: stage-based RAM snapshots
- `optimizer_steps`: per-step timing + samples + tokens
- `sequence_stats`: min/mean/max/p95 for input and target lengths
- `training_shape`: batch + grad accumulation + effective samples/step
- `checkpoint`: save time + size + file count
- `eval`: reload/eval/generation durations

