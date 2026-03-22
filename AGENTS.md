# AGENTS Guidance: Limit-Focused Instrumentation

When editing this repository, preserve and extend instrumentation first; avoid removing visibility unless broken.

## Instrumentation contract (must keep)
- Keep timestamped human-readable logs in CI output.
- Keep compact machine-readable metrics artifacts under `outputs/metrics/`.
- Keep per-optimizer-step timing and cumulative-sample reporting.
- Keep memory snapshots at the documented lifecycle stages.
- Keep sequence length and tokens-per-step statistics.
- Keep checkpoint timing and size reporting.
- Keep separate train/eval timing, including reload and generation timing when generation exists.
- Keep preprocessing/data-loader timing visibility.

## Change discipline
- Training behavior should remain unchanged unless instrumentation requires a minimal safe adjustment.
- Prefer low-overhead probes and aggregated stats over verbose raw payloads.
- Do not log full sample text, secrets, or huge config dumps.
- If adding new bottleneck probes, update `README.md` section **How to read the limit logs**.
- Any instrumentation removal requires a clear rationale in PR notes.

## Review checklist for future runs
A reviewer should be able to identify from one run whether the primary bottleneck is:
- runtime,
- RAM,
- token length,
- checkpoint I/O, or
- eval overhead.

If this cannot be determined from logs + `outputs/metrics/metrics-summary.json`, add instrumentation before tuning.
