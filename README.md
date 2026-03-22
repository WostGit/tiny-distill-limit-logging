# Tiny Distill Limit Logging Smoke Test

This repository contains a CPU-safe LoRA distillation smoke test with instrumentation focused on finding scaling limits quickly on `ubuntu-latest` GitHub Actions runners.

## Run

```bash
python train_smoke_distill.py --samples 64 --per-device-batch-size 2 --gradient-accumulation-steps 4
```

Metrics JSON is written to:

- `outputs/metrics/run_metrics.json`

## How to read the limit logs

The trainer emits timestamped, human-readable logs and a compact JSON artifact to classify which resource is likely to break first.

### 1) Runtime scaling limits
- Look for `optimizer_step=... step_duration=...` lines.
- If step duration increases over time, the likely bottleneck is cumulative compute or data overhead.
- Compare `tokenization/preprocessing duration` and `dataloader build duration` against training step timings to isolate front-loaded vs per-step costs.

### 2) Memory headroom limits
- Review `memory[...] rss=...` snapshots at process start, model load, LoRA wrap, first forward/backward, every optimizer step, save/eval boundaries.
- A steep jump at `after_first_forward_backward` indicates activation/gradient pressure.
- A steady climb at `optimizer_step_*` suggests retained tensors or fragmentation.

### 3) Token-length-driven limits
- Check sequence stats (`input` and `target` min/mean/max/p95).
- Compare `tokens_this_step` across optimizer steps.
- High p95 or high per-step token counts usually explain nonlinear runtime growth when sample count rises from 64 → 128 → 256.

### 4) Effective batch behavior
- Inspect the train-shape log for:
  - `per_device_batch_size`
  - `gradient_accumulation_steps`
  - `effective_samples_per_optimizer_step`
- This confirms whether throughput shifts are coming from true compute load or shape/accumulation configuration.

### 5) Checkpoint I/O limits
- Check `checkpoint save started` / `checkpoint save finished` and JSON checkpoint fields:
  - `save_duration_s`
  - `directory_size_bytes`
  - `file_count`
- If save duration is large relative to step duration, checkpoint I/O is a significant contributor.

### 6) Eval overhead limits
- Compare eval metrics in logs/JSON:
  - `reload_s`
  - `eval_forward_s`
  - `generation_s`
- If these become dominant at larger sample counts or more frequent evaluation, eval is the scaling limiter.

## Bottleneck diagnosis checklist

For a single CI run, you can classify likely limits quickly:

- **Runtime bound**: step duration dominates and grows with sample/token count.
- **RAM bound**: RSS is near runner memory ceilings or grows sharply after forward/backward.
- **Token-length bound**: p95 length and tokens/step are high relative to baseline.
- **Checkpoint I/O bound**: save duration is comparable to or larger than several train steps.
- **Eval bound**: reload/eval/generation times are disproportionately large.
