"""
Reusable safety checks for ReEngageEnv.

This file is additive.
It contains small standalone checks used by verifier and evaluation code.
"""

from typing import Any, Dict, List


VALID_ACTIONS = {0, 1, 2, 3, 4, 5}
CONTACT_ACTIONS = {0, 1, 2, 3, 4}
WAIT_ACTION = 5


def is_valid_action(action: int) -> bool:
    return action in VALID_ACTIONS


def is_contact_action(action: int) -> bool:
    return action in CONTACT_ACTIONS


def is_wait_action(action: int) -> bool:
    return action == WAIT_ACTION


def get_fatigue(observation: Dict[str, Any]) -> int:
    return int(observation.get("notification_fatigue", 0))


def fatigue_violation(
    action: int,
    observation: Dict[str, Any],
    fatigue_limit: int = 3,
) -> bool:
    return is_contact_action(action) and get_fatigue(observation) > fatigue_limit


def discount_abuse(
    action: int,
    observation: Dict[str, Any],
) -> bool:
    if action != 3:
        return False

    return (
        observation.get("ltv_bucket") == "low"
        and observation.get("discount_sensitivity") != "yes"
    )


def repeated_contact_spam(
    actions: List[int],
    max_repeats: int = 3,
) -> bool:
    repeated = 0
    last_action = None

    for action in actions:
        if is_contact_action(action):
            if action == last_action:
                repeated += 1
            else:
                repeated = 1

            if repeated > max_repeats:
                return True

            last_action = action
        else:
            repeated = 0
            last_action = None

    return False


def invalid_action_rate(actions: List[int]) -> float:
    if not actions:
        return 0.0

    invalid = sum(1 for action in actions if not is_valid_action(action))
    return invalid / len(actions)


def fatigue_violation_rate(
    transitions: List[Dict[str, Any]],
    fatigue_limit: int = 3,
) -> float:
    if not transitions:
        return 0.0

    violations = 0

    for transition in transitions:
        action = transition.get("action")
        observation = transition.get("observation", {})

        if fatigue_violation(action, observation, fatigue_limit):
            violations += 1

    return violations / len(transitions)