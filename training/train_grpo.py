"""
GRPO training scaffold for ReEngageEnv.

This file is additive.

Current status:
- Defines intended training entry point
- Checks that rollout/evaluation infrastructure exists
- Does NOT pretend to train yet

Next upgrade:
- Add TRL GRPOTrainer
- Add model loading
- Add reward function adapter
- Add rollout sampling
"""

from pathlib import Path


CONFIG_PATH = Path("training/config.yaml")
ROLLOUT_PATH = Path("training/rollout.py")
METRICS_PATH = Path("evaluation/metrics.py")
VERIFIER_PATH = Path("verifier/reward_verifier.py")


def check_training_prerequisites() -> None:
    required_files = [
        CONFIG_PATH,
        ROLLOUT_PATH,
        METRICS_PATH,
        VERIFIER_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required training prerequisites:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def main() -> None:
    check_training_prerequisites()

    print("[TRAINING SCAFFOLD READY]")
    print("Found config, rollout, metrics, and verifier.")
    print("Next step: integrate TRL GRPOTrainer.")


if __name__ == "__main__":
    main()