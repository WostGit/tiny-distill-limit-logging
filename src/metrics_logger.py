from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _human_seconds(value: float) -> str:
    return f"{value:.3f}s"


@dataclass
class LimitMetricsLogger:
    """Collects compact machine-readable metrics while emitting readable logs."""

    output_path: Path
    run_start: float = field(default_factory=time.perf_counter)
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.payload = {
            "created_at_utc": _utc_now_iso(),
            "events": [],
            "memory_snapshots": [],
            "step_metrics": [],
            "sequence_stats": {},
            "checkpoint": {},
            "train": {},
            "eval": {},
            "data_pipeline": {},
        }

    def elapsed(self) -> float:
        return time.perf_counter() - self.run_start

    def log(self, message: str) -> None:
        elapsed = self.elapsed()
        print(f"[{_utc_now_iso()} | +{_human_seconds(elapsed)}] {message}", flush=True)

    def add_event(self, name: str, **fields: Any) -> None:
        record = {"event": name, "elapsed_s": round(self.elapsed(), 6), **fields}
        self.payload["events"].append(record)

    def add_memory_snapshot(self, label: str, rss_mb: float, vms_mb: float | None = None) -> None:
        snapshot = {
            "label": label,
            "elapsed_s": round(self.elapsed(), 6),
            "rss_mb": round(rss_mb, 3),
        }
        if vms_mb is not None:
            snapshot["vms_mb"] = round(vms_mb, 3)
        self.payload["memory_snapshots"].append(snapshot)
        self.log(
            f"memory[{label}] rss={snapshot['rss_mb']}MB"
            + (f" vms={snapshot['vms_mb']}MB" if "vms_mb" in snapshot else "")
        )

    def add_step_metric(self, *, optimizer_step: int, step_duration_s: float, cumulative_samples: int, tokens_this_step: int) -> None:
        self.payload["step_metrics"].append(
            {
                "optimizer_step": optimizer_step,
                "elapsed_s": round(self.elapsed(), 6),
                "step_duration_s": round(step_duration_s, 6),
                "cumulative_samples": cumulative_samples,
                "tokens_this_step": tokens_this_step,
            }
        )

    def set_train_shape(self, **shape: Any) -> None:
        self.payload["train"].setdefault("shape", {}).update(shape)

    def set_sequence_stats(self, *, input_stats: dict[str, float], target_stats: dict[str, float]) -> None:
        self.payload["sequence_stats"] = {
            "input": input_stats,
            "target": target_stats,
        }

    def set_data_pipeline(self, **pipeline_metrics: Any) -> None:
        self.payload["data_pipeline"].update(pipeline_metrics)

    def set_checkpoint_metrics(self, **checkpoint_metrics: Any) -> None:
        self.payload["checkpoint"].update(checkpoint_metrics)

    def set_eval_metrics(self, **eval_metrics: Any) -> None:
        self.payload["eval"].update(eval_metrics)

    def write(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, sort_keys=True)
        self.log(f"wrote metrics artifact: {self.output_path}")
        return self.output_path


def directory_size_and_file_count(path: str | os.PathLike[str]) -> tuple[int, int]:
    total_size = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for filename in files:
            file_count += 1
            full_path = os.path.join(root, filename)
            total_size += os.path.getsize(full_path)
    return total_size, file_count
