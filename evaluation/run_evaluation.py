"""
Run multiple rollouts and compute evaluation metrics.
"""

import json
from typing import Any, Dict, List

from training.rollout import run_episode
from evaluation.metrics import summarize_results


def load_trajectory(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("trajectory", [])


def run_evaluation(num_runs: int = 5) -> None:
    results: List[Dict[str, Any]] = []
    trajectories: List[List[Dict[str, Any]]] = []

    print(f"[EVAL] Running {num_runs} episodes...\n")

    for i in range(num_runs):
        seed = 42 + i
        run_id = f"eval_run_{i}"

        result = run_episode(seed=seed, run_id=run_id)

        # add success flag
        result["success"] = result["total_reward"] > 0

        results.append(result)

        trajectory = load_trajectory(result["trajectory_path"])
        trajectories.append(trajectory)

    summary = summarize_results(results, trajectories)

    print("\n===== EVALUATION SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("==============================\n")


if __name__ == "__main__":
    run_evaluation(num_runs=5)