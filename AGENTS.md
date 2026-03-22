# AGENTS.md

Instrumentation discipline for this repository:

1. **Preserve bottleneck visibility.**
   - Do not remove per-optimizer-step timing logs, memory snapshots, or sequence length summaries.
2. **Extend metrics before changing behavior.**
   - If scaling confidence is low, add/adjust logging first; avoid changing training semantics unless required.
3. **Keep logs CI-readable.**
   - Prefer concise timestamped lines with key-value fields over large dumps.
   - Never print full sample text or giant configs.
4. **Always keep machine-readable metrics.**
   - Maintain JSON output under `outputs/metrics/`.
   - New instrumentation should be reflected in both console logs and JSON summary.
5. **Protect limit diagnostics.**
   - Ensure every run still provides enough signal to reason about runtime, RAM, token length, checkpoint IO, eval overhead, and data pipeline cost.
