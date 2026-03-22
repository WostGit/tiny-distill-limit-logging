from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = math.ceil((percentile / 100.0) * len(sorted_vals)) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "min": float(min(lengths)),
        "mean": float(statistics.fmean(lengths)),
        "max": float(max(lengths)),
        "p95": _percentile([float(v) for v in lengths], 95),
    }


def get_memory_snapshot() -> Dict[str, float]:
    if psutil is None:
        return {"rss_mb": -1.0, "vms_mb": -1.0, "available_mb": -1.0}
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    vm = psutil.virtual_memory()
    return {
        "rss_mb": round(mem.rss / (1024 * 1024), 2),
        "vms_mb": round(mem.vms / (1024 * 1024), 2),
        "available_mb": round(vm.available / (1024 * 1024), 2),
    }


@dataclass
class MetricsLogger:
    output_dir: str = "outputs/metrics"
    run_name: str = "distill_smoke"
    start_time: float = field(default_factory=time.perf_counter)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = {
            "run_name": self.run_name,
            "start_utc": _utc_now(),
            "events": [],
            "memory_snapshots": [],
            "optimizer_steps": [],
        }
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def log(self, message: str, **payload: Any) -> None:
        elapsed = round(time.perf_counter() - self.start_time, 3)
        ts = _utc_now()
        entry = {"timestamp_utc": ts, "elapsed_s": elapsed, "message": message, **payload}
        self.metrics["events"].append(entry)
        extras = " ".join(f"{k}={v}" for k, v in payload.items())
        print(f"[{ts}] +{elapsed:8.3f}s | {message}" + (f" | {extras}" if extras else ""), flush=True)

    def memory_snapshot(self, stage: str) -> None:
        snapshot = {"stage": stage, "timestamp_utc": _utc_now(), **get_memory_snapshot()}
        self.metrics["memory_snapshots"].append(snapshot)
        self.log("memory_snapshot", stage=stage, rss_mb=snapshot["rss_mb"], available_mb=snapshot["available_mb"])

    def add_optimizer_step(self, entry: Dict[str, Any]) -> None:
        self.metrics["optimizer_steps"].append(entry)
        self.log(
            "optimizer_step",
            step=entry["optimizer_step"],
            step_duration_s=entry["step_duration_s"],
            cumulative_samples=entry["cumulative_samples"],
            tokens_this_step=entry["tokens_this_step"],
        )

    def set(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def save(self) -> Path:
        end_utc = _utc_now()
        elapsed = round(time.perf_counter() - self.start_time, 3)
        self.metrics["end_utc"] = end_utc
        self.metrics["total_elapsed_s"] = elapsed
        stamped = Path(self.output_dir) / f"{self.run_name}_{int(time.time())}.json"
        latest = Path(self.output_dir) / "latest_metrics.json"
        payload = json.dumps(self.metrics, indent=2, sort_keys=True)
        stamped.write_text(payload + "\n", encoding="utf-8")
        latest.write_text(payload + "\n", encoding="utf-8")
        self.log("metrics_saved", path=str(stamped), total_elapsed_s=elapsed)
        return stamped
