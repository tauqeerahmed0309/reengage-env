"""
Simple demo app for ReEngageEnv.

Uses predefined scenarios and runs rollout.
"""

import json
from pathlib import Path

from training.rollout import run_episode


SCENARIO_PATH = Path("demo/scenarios.json")


def load_scenarios():
    with open(SCENARIO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["scenarios"]


def run_demo():
    scenarios = load_scenarios()

    print("\n===== ReEngageEnv Demo =====\n")

    for scenario in scenarios:
        print(f"Scenario: {scenario['title']}")
        print(f"Expected: {scenario['expected_behavior']}\n")

        result = run_episode(
            seed=scenario["seed"],
            run_id=scenario["id"],
        )

        print(f"Result: reward={result['total_reward']}, steps={result['steps']}")
        print("-" * 40)

    print("\n===== Demo Complete =====")


if __name__ == "__main__":
    run_demo()