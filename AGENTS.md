# AGENTS guidance for tiny-distill-limit-logging

Scope: entire repository.

## Instrumentation discipline (must preserve)

When changing training/eval/data code, preserve and extend limit-focused instrumentation rather than removing it.

Required observability areas:
1. Optimizer-step timing (elapsed since run start + per-step duration)
2. Memory snapshots at critical lifecycle boundaries
3. Sequence length stats (min/mean/max/p95) and tokens-per-step
4. Effective batch shape (per-device batch + gradient accumulation)
5. Checkpoint save timing and size/file-count metrics
6. Separate eval timing (reload, forward, generation when used)
7. Data pipeline timing (tokenization/preprocessing and dataloader overhead)

## Logging output expectations

- Keep logs human-readable for GitHub Actions.
- Keep runtime overhead low (lightweight timing/memory probes only).
- Avoid dumping full sample text or giant config blobs.
- Always write a compact JSON artifact under `outputs/metrics/`.

## Change policy

- Do not change core training behavior unless needed for instrumentation correctness.
- If behavior must change, document why and quantify impact in PR notes.
- Favor additive changes that increase confidence in scaling-limit diagnosis.
