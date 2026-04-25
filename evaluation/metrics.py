"""
Evaluation metrics for ReEngageEnv trajectories.

This file is additive.
It reads rollout trajectories and computes judge-facing metrics.
"""

from typing import Any, Dict, List


CONTACT_ACTIONS = {0, 1, 2, 3, 4}


def success_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0

    successes = sum(1 for r in results if r.get("success", False))
    return successes / len(results)


def average_reward(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0

    return sum(float(r.get("total_reward", 0.0)) for r in results) / len(results)


def average_steps(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0

    return sum(int(r.get("steps", 0)) for r in results) / len(results)


def verification_pass_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0

    passed = sum(1 for r in results if r.get("passed_verification", False))
    return passed / len(results)


def fatigue_violation_rate(trajectories: List[List[Dict[str, Any]]]) -> float:
    total = 0
    violations = 0

    for trajectory in trajectories:
        for transition in trajectory:
            total += 1
            action = transition.get("action")
            obs = transition.get("observation", {})
            fatigue = int(obs.get("notification_fatigue", 0))

            if action in CONTACT_ACTIONS and fatigue > 3:
                violations += 1

    if total == 0:
        return 0.0

    return violations / total


def channel_match_rate(trajectories: List[List[Dict[str, Any]]]) -> float:
    total_contacts = 0
    matches = 0

    channel_to_action = {
        "email": 0,
        "push": 1,
        "sms": 2,
    }

    for trajectory in trajectories:
        for transition in trajectory:
            action = transition.get("action")
            obs = transition.get("observation", {})
            pref = obs.get("channel_pref")

            if action in CONTACT_ACTIONS:
                total_contacts += 1
                if pref in channel_to_action and action == channel_to_action[pref]:
                    matches += 1

    if total_contacts == 0:
        return 0.0

    return matches / total_contacts


def discount_usage_rate(trajectories: List[List[Dict[str, Any]]]) -> float:
    total = 0
    discounts = 0

    for trajectory in trajectories:
        for transition in trajectory:
            total += 1
            if transition.get("action") == 3:
                discounts += 1

    if total == 0:
        return 0.0

    return discounts / total


def summarize_results(
    results: List[Dict[str, Any]],
    trajectories: List[List[Dict[str, Any]]],
) -> Dict[str, float]:
    return {
        "success_rate": round(success_rate(results), 4),
        "average_reward": round(average_reward(results), 4),
        "average_steps": round(average_steps(results), 4),
        "verification_pass_rate": round(verification_pass_rate(results), 4),
        "fatigue_violation_rate": round(fatigue_violation_rate(trajectories), 4),
        "channel_match_rate": round(channel_match_rate(trajectories), 4),
        "discount_usage_rate": round(discount_usage_rate(trajectories), 4),
    }