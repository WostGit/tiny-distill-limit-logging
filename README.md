# Tiny Distill Limit Logging Smoke Test

This repository contains a CPU-only tiny distillation smoke test with LoRA-style adapters and **limit-focused instrumentation** for GitHub Actions.

## Run locally

```bash
python train_smoke.py --train-samples 64 --batch-size 4 --grad-accum 4
```

The run emits human-readable timestamped logs and writes compact structured metrics to:

- `outputs/metrics/latest.json`
- `outputs/metrics/tiny-distill-<timestamp>.json`

## How to read the limit logs

Use this section to decide whether your next scaling limit is runtime, RAM, token length, checkpoint I/O, or eval overhead.

### 1) Runtime scaling limit

Look at `optimizer_step` events:
- `elapsed_since_run_start_s`
- `step_duration_s`
- `cumulative_samples`
- `tokens_in_optimizer_step`

If step duration climbs as token lengths increase, your next risk is compute/runtime saturation.

### 2) Memory headroom limit

Look at `memory` snapshots captured at:
- process start
- after model load
- after LoRA wrapping
- after first forward/backward
- every optimizer step
- before and after save
- before and after eval

If `max_rss_mb` approaches runner limits with higher sample counts/lengths, RAM is your next hard limit.

### 3) Token-length-driven cost limit

Look at `sequence_stats`:
- input min/mean/max/p95
- target min/mean/max/p95

And correlate with optimizer-step `tokens_in_optimizer_step`. High p95 lengths usually explain runtime jumps more than sample-count increases alone.

### 4) Effective batch behavior

Look at `train_shape`:
- `per_device_batch_size`
- `grad_accum_steps`
- `effective_samples_per_opt_step`

This clarifies whether pushes from 64 → 128 → 256 samples are stressing per-step compute or just total step count.

### 5) Checkpoint I/O limit

Look at:
- `checkpoint_save_start`
- `checkpoint_save_end.save_duration_s`
- `checkpoint_save_end.checkpoint_bytes`
- `checkpoint_save_end.checkpoint_file_count`

If save duration grows disproportionately, checkpoint I/O is becoming a bottleneck.

### 6) Eval overhead limit

Look at `eval`:
- `reload_duration_s`
- `eval_duration_s`
- `generation_duration_s`

If eval+reload is a large fraction of total runtime, scaling train samples may not be your immediate limit.

### 7) Data pipeline overhead

Look at:
- `preprocess.duration_s`
- per-batch `dataloader.duration_s`

If dataloader/preprocessing time is non-trivial compared to step time, optimize data first before scaling dataset size.

## Why 64 likely safe, 128 plausible, 256 stretched

This instrumentation lets you test that hypothesis directly:
1. Run 64/128/256 with same batch+accum.
2. Compare p95 lengths, step durations, and memory peaks.
3. Check whether save/eval overhead becomes a significant runtime share.

The safest dimension to push next is whichever increases total runtime while keeping memory plateau and checkpoint/eval overhead stable.
