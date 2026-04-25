"""
Independent reward and trajectory verifier for ReEngageEnv.

This file is additive.
It does not modify the environment.
It checks whether environment transitions look valid, safe, and non-exploitative.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_ACTIONS = {0, 1, 2, 3, 4, 5}

ACTION_NAMES = {
    0: "email",
    1: "push",
    2: "sms",
    3: "discount",
    4: "feature_tip",
    5: "wait",
}

CONTACT_ACTIONS = {0, 1, 2, 3, 4}
WAIT_ACTION = 5


@dataclass
class VerificationIssue:
    severity: str
    code: str
    message: str
    step: Optional[int] = None


@dataclass
class VerificationResult:
    passed: bool
    issues: List[VerificationIssue] = field(default_factory=list)

    def add_issue(
        self,
        severity: str,
        code: str,
        message: str,
        step: Optional[int] = None,
    ) -> None:
        self.issues.append(
            VerificationIssue(
                severity=severity,
                code=code,
                message=message,
                step=step,
            )
        )
        if severity in {"error", "critical"}:
            self.passed = False


class RewardVerifier:
    """
    Verifies individual transitions and full trajectories.

    This is not the reward function.
    This is a separate safety and consistency layer.
    """

    def __init__(
        self,
        min_reward: float = -1.0,
        max_reward: float = 2.5,
        fatigue_limit: int = 3,
        max_repeated_contact: int = 3,
    ) -> None:
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.fatigue_limit = fatigue_limit
        self.max_repeated_contact = max_repeated_contact

    def verify_action(self, action: int, step: Optional[int] = None) -> VerificationResult:
        result = VerificationResult(passed=True)

        if action not in VALID_ACTIONS:
            result.add_issue(
                severity="critical",
                code="invalid_action",
                message=f"Action {action} is invalid. Expected one of {sorted(VALID_ACTIONS)}.",
                step=step,
            )

        return result

    def verify_reward_range(
        self,
        reward: float,
        step: Optional[int] = None,
    ) -> VerificationResult:
        result = VerificationResult(passed=True)

        if reward < self.min_reward or reward > self.max_reward:
            result.add_issue(
                severity="error",
                code="reward_out_of_range",
                message=f"Reward {reward} is outside expected range [{self.min_reward}, {self.max_reward}].",
                step=step,
            )

        return result

    def verify_fatigue(
        self,
        action: int,
        observation: Dict[str, Any],
        step: Optional[int] = None,
    ) -> VerificationResult:
        result = VerificationResult(passed=True)

        fatigue = observation.get("notification_fatigue")

        if fatigue is None:
            result.add_issue(
                severity="warning",
                code="missing_fatigue",
                message="Observation does not contain notification_fatigue.",
                step=step,
            )
            return result

        if action in CONTACT_ACTIONS and fatigue > self.fatigue_limit:
            result.add_issue(
                severity="error",
                code="fatigue_violation",
                message=(
                    f"Contact action '{ACTION_NAMES[action]}' used when "
                    f"notification_fatigue={fatigue}, above safe limit {self.fatigue_limit}."
                ),
                step=step,
            )

        return result

    def verify_discount_usage(
        self,
        action: int,
        observation: Dict[str, Any],
        step: Optional[int] = None,
    ) -> VerificationResult:
        result = VerificationResult(passed=True)

        if action != 3:
            return result

        ltv_bucket = observation.get("ltv_bucket")
        discount_sensitivity = observation.get("discount_sensitivity")

        if ltv_bucket == "low" and discount_sensitivity != "yes":
            result.add_issue(
                severity="warning",
                code="weak_discount_justification",
                message=(
                    "Discount sent to low-LTV user without clear discount sensitivity. "
                    "This may waste promo budget."
                ),
                step=step,
            )

        return result

    def verify_transition(
        self,
        action: int,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> VerificationResult:
        result = VerificationResult(passed=True)

        checks = [
            self.verify_action(action, step),
            self.verify_reward_range(reward, step),
            self.verify_fatigue(action, observation, step),
            self.verify_discount_usage(action, observation, step),
        ]

        for check in checks:
            result.issues.extend(check.issues)
            if not check.passed:
                result.passed = False

        if info is None:
            result.add_issue(
                severity="warning",
                code="missing_info",
                message="Step response does not include info dictionary.",
                step=step,
            )

        if done and info:
            if "re_engaged" not in info:
                result.add_issue(
                    severity="warning",
                    code="missing_reengagement_flag",
                    message="Episode ended but info does not include re_engaged flag.",
                    step=step,
                )

        return result

    def verify_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> VerificationResult:
        result = VerificationResult(passed=True)

        repeated_contact_count = 0
        last_contact_action = None

        for idx, transition in enumerate(trajectory):
            step = transition.get("step", idx + 1)
            action = transition.get("action")
            observation = transition.get("observation", {})
            reward = transition.get("reward", 0.0)
            done = transition.get("done", False)
            info = transition.get("info", {})

            transition_result = self.verify_transition(
                action=action,
                observation=observation,
                reward=reward,
                done=done,
                info=info,
                step=step,
            )

            result.issues.extend(transition_result.issues)
            if not transition_result.passed:
                result.passed = False

            if action in CONTACT_ACTIONS:
                if action == last_contact_action:
                    repeated_contact_count += 1
                else:
                    repeated_contact_count = 1

                last_contact_action = action

                if repeated_contact_count > self.max_repeated_contact:
                    result.add_issue(
                        severity="error",
                        code="repeated_contact_spam",
                        message=(
                            f"Same contact action '{ACTION_NAMES[action]}' repeated "
                            f"{repeated_contact_count} times."
                        ),
                        step=step,
                    )
            else:
                repeated_contact_count = 0
                last_contact_action = None

        return result


if __name__ == "__main__":
    verifier = RewardVerifier()

    sample_transition = {
        "step": 1,
        "action": 0,
        "observation": {
            "notification_fatigue": 4,
            "ltv_bucket": "medium",
            "discount_sensitivity": "unknown",
        },
        "reward": 0.18,
        "done": False,
        "info": {"re_engaged": False},
    }

    result = verifier.verify_trajectory([sample_transition])

    print("Passed:", result.passed)
    for issue in result.issues:
        print(issue)