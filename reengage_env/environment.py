"""
ReEngageEnv — Re-engagement RL Environment
OpenEnv-compliant environment for training agents to re-engage dormant users
via optimal notification strategy.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

DropOffPoint      = Literal["onboarding", "dashboard", "checkout", "settings", "other"]
SubscriptionStatus= Literal["free", "trial", "paid", "cancelled"]
LTVBucket         = Literal["low", "medium", "high"]
DeviceType        = Literal["mobile", "desktop", "tablet"]
ChannelPref       = Literal["email", "push", "sms", "none"]
UserStatus        = Literal["new", "active", "dormant", "churned"]
BehavioralCluster = Literal["browser", "buyer", "lurker", "power_user"]
DiscountSensitivity = Literal["yes", "no", "unknown"]
ActionName        = Literal["email", "push", "sms", "discount", "feature_tip", "wait"]


class Observation(BaseModel):
    """Full observation returned by step() and reset()."""

    # --- User Behavior Features ---
    days_since_active:      int   = Field(..., ge=0, le=365,   description="Days since user last had a session")
    session_freq_7d:        int   = Field(..., ge=0, le=7,     description="Sessions in the last 7 days")
    session_freq_30d:       int   = Field(..., ge=0, le=30,    description="Sessions in the last 30 days")
    avg_session_dur:        float = Field(..., ge=0.0, le=120.0, description="Average session duration in minutes")
    features_used:          int   = Field(..., ge=0, le=10,    description="Count of distinct product features touched")
    drop_off_point:         DropOffPoint                       = Field(..., description="Last screen/stage where user dropped off")

    # --- Value / Monetization Features ---
    total_spend:            float = Field(..., ge=0.0, le=10000.0, description="Total USD spend lifetime")
    avg_order_value:        float = Field(..., ge=0.0, le=500.0,   description="Average order value in USD")
    subscription_status:   SubscriptionStatus                 = Field(..., description="Current subscription tier")
    ltv_bucket:             LTVBucket                         = Field(..., description="Lifetime value segment")

    # --- Engagement Context ---
    time_of_day:            int   = Field(..., ge=0, le=23,    description="Current hour in UTC")
    day_of_week:            int   = Field(..., ge=0, le=6,     description="Day of week, Monday=0")
    device_type:            DeviceType                        = Field(..., description="User's primary device type")
    notification_fatigue:   int   = Field(..., ge=0, le=10,    description="Messages already sent this week")
    channel_pref:           ChannelPref                       = Field(..., description="User's preferred contact channel")

    # --- User Segmentation ---
    user_status:            UserStatus                        = Field(..., description="High-level activity status")
    behavioral_cluster:     BehavioralCluster                 = Field(..., description="Behavioral archetype cluster")
    discount_sensitivity:   DiscountSensitivity               = Field(..., description="Whether user responds to discounts")

    # --- Action History ---
    last_action:            ActionName                        = Field(..., description="Last action taken by agent")
    num_attempts:           int   = Field(..., ge=0, le=50,    description="Total outreach attempts this episode")
    days_since_last_notif:  int   = Field(..., ge=0, le=365,   description="Days since most recent notification")


class Action(BaseModel):
    """Action input validated by the environment."""
    action_id: int = Field(..., ge=0, le=5, description="0=email,1=push,2=sms,3=discount,4=feature_tip,5=wait")


class RewardBreakdown(BaseModel):
    """Detailed breakdown of the reward signal for a single step."""
    re_engagement:   float
    channel_match:   float
    fatigue_penalty: float
    discount_penalty:float
    wait_bonus:      float
    step_cost:       float
    total:           float


class EnvState(BaseModel):
    """Full internal state — available to graders, NOT to agents during training."""
    observation:          Observation
    step:                 int
    done:                 bool
    re_engaged:           bool
    cumulative_reward:    float
    churn_probability:    float   = Field(..., description="Hidden true churn probability [0,1]")
    last_reward_breakdown:Optional[RewardBreakdown]


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

ACTION_NAMES: Dict[int, ActionName] = {
    0: "email",
    1: "push",
    2: "sms",
    3: "discount",
    4: "feature_tip",
    5: "wait",
}

CHANNEL_TO_ACTION: Dict[str, int] = {
    "email": 0,
    "push":  1,
    "sms":   2,
}

LTV_MULTIPLIER: Dict[str, float] = {
    "low":    1.0,
    "medium": 1.5,
    "high":   2.0,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ReEngageEnv:
    """
    OpenEnv-compliant Re-engagement RL Environment.

    Observation space: 20-field Pydantic Observation model
    Action space:      Discrete(6) — 0=email, 1=push, 2=sms,
                                     3=discount, 4=feature_tip, 5=wait
    Reward range:      [-1.0, 2.0] shaped per step
    Episode length:    max_steps (default 10)

    Usage
    -----
        env = ReEngageEnv(seed=42)
        obs = env.reset()
        obs, reward, done, info = env.step(0)   # send email
        state = env.state()                      # full internal state (grader use)
    """

    # Sampling pools for random user generation
    _STATUSES:  list[UserStatus]        = ["new", "active", "dormant", "churned"]
    _CLUSTERS:  list[BehavioralCluster] = ["browser", "buyer", "lurker", "power_user"]
    _DROP_OFFS: list[DropOffPoint]      = ["onboarding", "dashboard", "checkout", "settings", "other"]
    _SUB_STATS: list[SubscriptionStatus]= ["free", "trial", "paid", "cancelled"]
    _DEVICES:   list[DeviceType]        = ["mobile", "desktop", "tablet"]
    _CHANNELS:  list[ChannelPref]       = ["email", "push", "sms", "none"]
    _DISC_SENS: list[DiscountSensitivity]= ["yes", "no", "unknown"]

    def __init__(self, max_steps: int = 10, seed: Optional[int] = None):
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._obs: Optional[Observation] = None
        self._step = 0
        self._done = False
        self._re_engaged = False
        self._cumulative_reward = 0.0
        self._churn_probability = 0.0
        self._last_breakdown: Optional[RewardBreakdown] = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Sample a new user and return the initial observation."""
        self._obs = self._sample_user()
        self._step = 0
        self._done = False
        self._re_engaged = False
        self._cumulative_reward = 0.0
        self._last_breakdown = None
        self._churn_probability = self._compute_churn_prob(self._obs)
        return self._obs

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply action and advance the environment one step.

        Parameters
        ----------
        action : int
            Action id in [0, 5].

        Returns
        -------
        observation : Observation
        reward      : float
        done        : bool
        info        : dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        Action(action_id=action)  # validate
        action_name: ActionName = ACTION_NAMES[action]

        reward, breakdown = self._compute_reward(action_name)
        self._update_state(action_name)
        self._step += 1
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)

        if self._re_engaged or self._step >= self.max_steps:
            self._done = True

        self._last_breakdown = breakdown

        info: Dict[str, Any] = {
            "re_engaged":          self._re_engaged,
            "step":                self._step,
            "done":                self._done,
            "cumulative_reward":   self._cumulative_reward,
            "reward_breakdown":    breakdown.model_dump(),
            "notification_fatigue":self._obs.notification_fatigue,
        }

        return self._obs, reward, self._done, info

    def state(self) -> EnvState:
        """Return full internal state (for graders — do not expose to agents)."""
        if self._obs is None:
            raise RuntimeError("Call reset() first.")
        return EnvState(
            observation=self._obs,
            step=self._step,
            done=self._done,
            re_engaged=self._re_engaged,
            cumulative_reward=self._cumulative_reward,
            churn_probability=self._churn_probability,
            last_reward_breakdown=self._last_breakdown,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_user(self) -> Observation:
        r = self._rng
        status: UserStatus = r.choice(self._STATUSES)

        # Days since active varies by status
        days_map = {"new": (0, 2), "active": (0, 3), "dormant": (7, 45), "churned": (45, 180)}
        lo, hi = days_map[status]
        days_since_active = r.randint(lo, hi)

        session_freq_7d  = r.randint(0, 5) if status == "active" else 0
        session_freq_30d = r.randint(0, 10) if status in ("active", "new") else r.randint(0, 3)

        ltv: LTVBucket = r.choice(["low", "medium", "high"])
        spend_map = {"low": (0.0, 49.0), "medium": (50.0, 499.0), "high": (500.0, 5000.0)}
        total_spend = round(r.uniform(*spend_map[ltv]), 2)
        avg_order = round(min(500.0, total_spend / max(1, r.randint(1, 10))), 2)

        return Observation(
            # Behavior
            days_since_active  = days_since_active,
            session_freq_7d    = session_freq_7d,
            session_freq_30d   = session_freq_30d,
            avg_session_dur    = round(r.uniform(1.0, 45.0), 1),
            features_used      = r.randint(0, 10),
            drop_off_point     = r.choice(self._DROP_OFFS),
            # Value
            total_spend        = total_spend,
            avg_order_value    = avg_order,
            subscription_status= r.choice(self._SUB_STATS),
            ltv_bucket         = ltv,
            # Context
            time_of_day        = r.randint(0, 23),
            day_of_week        = r.randint(0, 6),
            device_type        = r.choice(self._DEVICES),
            notification_fatigue = r.randint(0, 4),
            channel_pref       = r.choice(self._CHANNELS),
            # Segmentation
            user_status        = status,
            behavioral_cluster = r.choice(self._CLUSTERS),
            discount_sensitivity = r.choice(self._DISC_SENS),
            # Action history (fresh episode)
            last_action        = "wait",
            num_attempts       = 0,
            days_since_last_notif = r.randint(1, 60),
        )

    def _compute_churn_prob(self, obs: Observation) -> float:
        """Hidden ground-truth churn probability used to modulate re-engagement."""
        p = 0.0
        p += obs.days_since_active / 180.0 * 0.5
        if obs.user_status == "churned":   p += 0.3
        elif obs.user_status == "dormant": p += 0.15
        p += obs.notification_fatigue / 10.0 * 0.2
        if obs.behavioral_cluster == "buyer": p -= 0.1
        return min(1.0, max(0.0, round(p, 3)))

    def _compute_reward(self, action_name: ActionName) -> Tuple[float, RewardBreakdown]:
        obs = self._obs

        step_cost      = -0.02
        re_eng_reward  = 0.0
        channel_match  = 0.0
        fatigue_penalty= 0.0
        discount_pen   = 0.0
        wait_bonus     = 0.0

        if action_name == "wait":
            if obs.notification_fatigue >= 3:
                wait_bonus = 0.05
        else:
            # Fatigue penalty
            if obs.notification_fatigue >= 4:
                fatigue_penalty = -0.15 * (obs.notification_fatigue - 3)

            # Channel-match bonus
            if obs.channel_pref != "none":
                expected_action = CHANNEL_TO_ACTION.get(obs.channel_pref)
                if expected_action is not None:
                    test_action = CHANNEL_TO_ACTION.get(action_name, -1)
                    if test_action == expected_action:
                        channel_match = 0.2

            # Discount penalty for low-LTV users
            if action_name == "discount" and obs.ltv_bucket == "low":
                discount_pen = -0.1

            # Re-engagement probability (stochastic)
            p_reengage = self._compute_reengage_prob(action_name)
            if self._rng.random() < p_reengage:
                ltv_mult   = LTV_MULTIPLIER[obs.ltv_bucket]
                re_eng_reward = 1.0 * ltv_mult
                self._re_engaged = True

        total = round(
            step_cost + re_eng_reward + channel_match + fatigue_penalty + discount_pen + wait_bonus,
            4,
        )

        breakdown = RewardBreakdown(
            re_engagement   = re_eng_reward,
            channel_match   = channel_match,
            fatigue_penalty = fatigue_penalty,
            discount_penalty= discount_pen,
            wait_bonus      = wait_bonus,
            step_cost       = step_cost,
            total           = total,
        )
        return total, breakdown

    def _compute_reengage_prob(self, action_name: ActionName) -> float:
        """
        Deterministic function of observation + action → re-engagement probability.
        Reproducible given the same RNG seed (used to sample the outcome, not here).
        """
        obs = self._obs
        p = 0.05

        # Recency factor: recent inactivity is easier to recover
        recency = max(0.0, 1.0 - obs.days_since_active / 90.0)
        p += recency * 0.25

        # Channel match strongly boosts probability
        if obs.channel_pref != "none":
            action_chan = CHANNEL_TO_ACTION.get(action_name)
            pref_chan   = CHANNEL_TO_ACTION.get(obs.channel_pref)
            if action_chan is not None and action_chan == pref_chan:
                p += 0.25

        # Discount works for sensitive users, hurts for resistant ones
        if action_name == "discount":
            if obs.discount_sensitivity == "yes":
                p += 0.20
            elif obs.discount_sensitivity == "no":
                p -= 0.10

        # Buyer cluster is more likely to return
        if obs.behavioral_cluster == "buyer":
            p += 0.10
        elif obs.behavioral_cluster == "lurker":
            p -= 0.05

        # Fatigue tanks probability
        if obs.notification_fatigue >= 5:
            p -= 0.30
        elif obs.notification_fatigue >= 3:
            p -= 0.10

        # Feature tip works well for power users
        if action_name == "feature_tip" and obs.behavioral_cluster == "power_user":
            p += 0.15

        return min(0.95, max(0.0, p))

    def _update_state(self, action_name: ActionName) -> None:
        """Mutate observation fields after taking an action."""
        obs = self._obs

        # Rebuild observation as a dict, mutate, then re-parse
        d = obs.model_dump()

        d["last_action"] = action_name

        if action_name != "wait":
            d["notification_fatigue"]  = min(10, obs.notification_fatigue + 1)
            d["num_attempts"]          = obs.num_attempts + 1
            d["days_since_last_notif"] = 0
        else:
            # Natural time passes: fatigue decays, days tick up
            d["notification_fatigue"]  = max(0, obs.notification_fatigue - 1)
            d["days_since_last_notif"] = min(365, obs.days_since_last_notif + 1)

        if self._re_engaged:
            d["days_since_active"]  = 0
            d["session_freq_7d"]    = min(7, obs.session_freq_7d + 1)
            d["session_freq_30d"]   = min(30, obs.session_freq_30d + 1)
            d["user_status"]        = "active"

        self._obs = Observation(**d)
