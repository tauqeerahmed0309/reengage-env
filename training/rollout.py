"""
Rollout collection for ReEngageEnv.

Offline version:
- no OpenAI API required
- uses deterministic policy
- connects environment server
- runs verifier
- writes audit + trajectory logs
"""

import os
from typing import Any, Dict, List

import requests

from verifier.reward_verifier import RewardVerifier
from verifier.audit_logger import AuditLogger


ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MAX_STEPS = 10

verifier = RewardVerifier()
logger = AuditLogger()


ACTION_NAMES = {
    0: "email",
    1: "push",
    2: "sms",
    3: "discount",
    4: "feature_tip",
    5: "wait",
}


def get_action_from_policy(obs: Dict[str, Any]) -> int:
    """
    Deterministic baseline policy.

    Rules:
    - If fatigue is high, wait.
    - If user prefers email/push/sms, use that channel.
    - If high-LTV and discount sensitive, send discount.
    - Otherwise send feature tip.
    """

    fatigue = int(obs.get("notification_fatigue", 0))
    channel_pref = obs.get("channel_pref", "none")
    ltv_bucket = obs.get("ltv_bucket", "low")
    discount_sensitivity = obs.get("discount_sensitivity", "unknown")

    if fatigue >= 3:
        return 5  # wait

    if ltv_bucket == "high" and discount_sensitivity == "yes":
        return 3  # discount

    channel_mapping = {
        "email": 0,
        "push": 1,
        "sms": 2,
    }

    if channel_pref in channel_mapping:
        return channel_mapping[channel_pref]

    return 4  # feature_tip


def reset_env(seed: int) -> Dict[str, Any]:
    response = requests.post(f"{ENV_URL}/reset", json={"seed": seed}, timeout=10)
    response.raise_for_status()
    return response.json()


def step_env(action: int) -> Dict[str, Any]:
    response = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=10)
    response.raise_for_status()
    return response.json()


def run_episode(seed: int = 42, run_id: str = "debug_run") -> Dict[str, Any]:
    obs = reset_env(seed)

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0

    print(f"[START] run_id={run_id} seed={seed}")

    for step in range(1, MAX_STEPS + 1):
        action = get_action_from_policy(obs)

        step_result = step_env(action)

        reward = float(step_result.get("reward", 0.0))
        done = bool(step_result.get("done", False))
        info = step_result.get("info", {})
        next_obs = step_result.get("observation", obs)

        transition = {
            "step": step,
            "action": action,
            "action_name": ACTION_NAMES.get(action, "unknown"),
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }

        trajectory.append(transition)

        verification = verifier.verify_transition(
            action=action,
            observation=obs,
            reward=reward,
            done=done,
            info=info,
            step=step,
        )

        if not verification.passed or verification.issues:
            logger.log_issues(
                run_id=run_id,
                issues=verification.issues,
                metadata={"step": step},
            )

        total_reward += reward

        print(
            f"[STEP] step={step} "
            f"action={action}:{ACTION_NAMES.get(action)} "
            f"reward={reward:.2f} "
            f"done={done}"
        )

        obs = next_obs

        if done:
            break

    final_check = verifier.verify_trajectory(trajectory)

    if not final_check.passed or final_check.issues:
        logger.log_issues(
            run_id=run_id,
            issues=final_check.issues,
            metadata={"phase": "final"},
        )

    trajectory_path = logger.log_trajectory(
        run_id=run_id,
        trajectory=trajectory,
        metadata={
            "seed": seed,
            "total_reward": total_reward,
            "steps": len(trajectory),
            "passed_verification": final_check.passed,
        },
    )

    result = {
        "run_id": run_id,
        "seed": seed,
        "total_reward": round(total_reward, 4),
        "steps": len(trajectory),
        "passed_verification": final_check.passed,
        "trajectory_path": str(trajectory_path),
    }

    print(f"[END] {result}")
    return result


if __name__ == "__main__":
    run_episode(seed=42, run_id="offline_debug_run")