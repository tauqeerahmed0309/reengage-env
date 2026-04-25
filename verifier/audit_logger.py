"""
Audit logger for ReEngageEnv verifier outputs.

This file is additive.
It stores suspicious trajectories and verification issues for later inspection.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class AuditLogger:
    def __init__(self, log_dir: str = "experiments/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_issues(
        self,
        run_id: str,
        issues: List[Any],
        metadata: Dict[str, Any] | None = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).isoformat()
        path = self.log_dir / f"{run_id}_audit.jsonl"

        record = {
            "timestamp": timestamp,
            "run_id": run_id,
            "metadata": metadata or {},
            "issues": [
                issue.__dict__ if hasattr(issue, "__dict__") else str(issue)
                for issue in issues
            ],
        }

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return path

    def log_trajectory(
        self,
        run_id: str,
        trajectory: List[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).isoformat()
        path = self.log_dir / f"{run_id}_trajectory.json"

        record = {
            "timestamp": timestamp,
            "run_id": run_id,
            "metadata": metadata or {},
            "trajectory": trajectory,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return path 