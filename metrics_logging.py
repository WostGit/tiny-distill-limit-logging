import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricsLogger:
    run_name: str
    output_dir: Path = Path("outputs/metrics")
    start_time: float = field(default_factory=time.perf_counter)
    events: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def record(self, event: str, **payload: Any) -> None:
        data = {
            "event": event,
            "elapsed_s": round(self.elapsed_seconds(), 4),
            **payload,
        }
        self.events.append(data)
        logging.info("[metrics] %s", json.dumps(data, sort_keys=True))

    def set_summary(self, **payload: Any) -> None:
        self.summary.update(payload)

    def write(self) -> Path:
        output = {
            "run_name": self.run_name,
            "started_at_unix": self.start_time,
            "summary": self.summary,
            "events": self.events,
        }
        ts = int(time.time())
        run_path = self.output_dir / f"{self.run_name}-{ts}.json"
        latest_path = self.output_dir / "latest.json"
        run_path.write_text(json.dumps(output, indent=2, sort_keys=True))
        latest_path.write_text(json.dumps(output, indent=2, sort_keys=True))
        logging.info("[metrics] wrote metrics artifact to %s", run_path)
        return run_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def dir_size_and_files(path: Path) -> Dict[str, int]:
    total_bytes = 0
    file_count = 0
    if not path.exists():
        return {"bytes": 0, "file_count": 0}
    for root, _, files in os.walk(path):
        for name in files:
            file_count += 1
            total_bytes += (Path(root) / name).stat().st_size
    return {"bytes": total_bytes, "file_count": file_count}
