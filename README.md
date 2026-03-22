# Tiny Distill Limit Logging Smoke Test

This repository provides a CPU-safe tiny distillation smoke test with LoRA-style adapter training and **limit-focused instrumentation** intended for GitHub Actions logs.

## Run

```bash
python train_distill_smoke.py --samples 64 --batch-size 4 --grad-accum 2
```

The run writes machine-readable metrics to:

- `outputs/metrics/metrics-summary.json`

## How to read the limit logs

The training script emits timestamped line logs in this format:

- `[timestamp] [+elapsed_s] event key=value ...`

Use these signals to diagnose bottlenecks:

1. **Runtime scaling ceiling**
   - Check `optimizer_step` events (`step_duration_s`) for per-step growth.
   - Check `train_complete duration_s` and `train_duration_s` metric.
   - If per-step time grows with sample count while memory is stable, runtime is likely your next limit.

2. **RAM headroom**
   - Inspect `memory_snapshot` events at key stages:
     - process start
     - model load
     - LoRA wrap
     - first forward/backward
     - every optimizer step
     - before/after save
     - before/after eval
   - If RSS steadily trends upward by step, memory pressure is likely.

3. **Token-length-driven cost**
   - Read `sequence_stats` event and JSON fields:
     - `input_len_stats` (min/mean/max/p95)
     - `target_len_stats` (min/mean/max/p95)
     - `tokens_per_optimizer_step`
   - If p95/max token lengths and tokens/step rise, expect step time growth even at fixed sample count.

4. **Effective batch behavior**
   - Read `training_shape` event/metric:
     - `per_device_batch_size`
     - `gradient_accumulation_steps`
     - `effective_samples_per_optimizer_step`
   - Use this to reason about why 64 is typically safe, 128 plausible, and 256 stretched on CPU.

5. **Checkpoint I/O overhead**
   - Inspect `checkpoint_save_start` / `checkpoint_save_end` events.
   - Confirm `checkpoint.save_duration_s`, `checkpoint.size_mb`, and `checkpoint.file_count` in metrics JSON.
   - If save time rises sharply with adapter size, checkpoint I/O is becoming a limit.

6. **Eval overhead**
   - Inspect `eval_complete` event and `eval` metrics:
     - `reload_duration_s`
     - `eval_duration_s`
     - `generation_duration_s`
   - If eval dominates wall-clock, reduce eval cadence before increasing dataset size.

7. **Data pipeline overhead**
   - Inspect preprocessing + loader metrics:
     - `preprocessing_time_s`
     - `dataloader_build_time_s`
     - `first_batch_load_time_s`
   - If these dominate total runtime, optimize tokenization/collation before changing model settings.

