import json
import math
import os
import resource
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rss_bytes() -> int:
    proc_status = Path("/proc/self/status")
    if proc_status.exists():
        for line in proc_status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    # Fallback for non-linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    vals = sorted(values)
    k = (len(vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


@dataclass
class MetricsLogger:
    output_dir: Path
    run_name: str = "distill-smoke"
    started_at_monotonic: float = field(default_factory=time.perf_counter)
    started_at_iso: str = field(default_factory=_now_iso)
    events: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def log_event(self, event: str, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self.started_at_monotonic
        payload = {
            "event": event,
            "timestamp": _now_iso(),
            "elapsed_s": round(elapsed, 3),
            **kwargs,
        }
        self.events.append(payload)
        kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[{payload['timestamp']}] [+{payload['elapsed_s']:.3f}s] {event} {kv}".rstrip())

    def snapshot_memory(self, stage: str) -> None:
        rss = _rss_bytes()
        self.log_event("memory_snapshot", stage=stage, rss_mb=round(rss / (1024 * 1024), 2))

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    @staticmethod
    def sequence_stats(lengths: list[int]) -> dict[str, float]:
        if not lengths:
            return {"min": 0, "mean": 0, "max": 0, "p95": 0}
        return {
            "min": int(min(lengths)),
            "mean": round(float(statistics.mean(lengths)), 3),
            "max": int(max(lengths)),
            "p95": round(_percentile([float(x) for x in lengths], 0.95), 3),
        }

    @staticmethod
    def directory_size(path: Path) -> tuple[int, int]:
        total_bytes = 0
        file_count = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = Path(root) / f
                if fp.exists():
                    total_bytes += fp.stat().st_size
                    file_count += 1
        return total_bytes, file_count

    def write_summary(self, filename: str = "metrics-summary.json") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        payload = {
            "run_name": self.run_name,
            "started_at": self.started_at_iso,
            "finished_at": _now_iso(),
            "duration_s": round(time.perf_counter() - self.started_at_monotonic, 3),
            "metrics": self.metrics,
            "events": self.events,
        }
        path.write_text(json.dumps(payload, indent=2))
        self.log_event("metrics_written", path=str(path))
        return path
