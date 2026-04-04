"""
Graders for ReEngageEnv tasks.

Each grader is a pure function:
    grade(env, task_kwargs) -> float in [0.0, 1.0]

All graders are deterministic and reproducible given the same seed.
"""

from __future__ import annotations

from typing import Any, Dict

from reengage_env.environment import ReEngageEnv


# ---------------------------------------------------------------------------
# Task 1 — Re-engage a dormant user (Easy)
# ---------------------------------------------------------------------------

def grade_task1(seed: int = 0, verbose: bool = False) -> float:
    """
    Task 1: Re-engage a dormant user within 5 steps.

    Scoring:
      - Re-engaged in ≤ 3 steps → 1.0
      - Re-engaged in ≤ 5 steps → 0.5
      - Not re-engaged           → 0.0

    Agent policy: channel-matched action if preference set, else email.
    """
    env = ReEngageEnv(max_steps=5, seed=seed)
    obs = env.reset()

    # Force a dormant user with known channel preference for determinism
    # (reset samples randomly; we loop until we hit a dormant user)
    attempts = 0
    while obs.user_status not in ("dormant",) and attempts < 20:
        obs = env.reset()
        attempts += 1

    from reengage_env.environment import CHANNEL_TO_ACTION

    score = 0.0
    for step_num in range(1, 6):
        # Simple channel-matching heuristic agent
        if obs.channel_pref != "none" and obs.channel_pref in CHANNEL_TO_ACTION:
            action = CHANNEL_TO_ACTION[obs.channel_pref]
        else:
            action = 0  # default: email

        obs, reward, done, info = env.step(action)
        if verbose:
            print(f"  Step {step_num}: action={info.get('reward_breakdown',{})}, reward={reward:.3f}, re_engaged={info['re_engaged']}")

        if info["re_engaged"]:
            score = 1.0 if step_num <= 3 else 0.5
            break

    if verbose:
        print(f"Task 1 score: {score:.2f}")
    return score


# ---------------------------------------------------------------------------
# Task 2 — Win-back a high-value churned user without spamming (Medium)
# ---------------------------------------------------------------------------

def grade_task2(seed: int = 0, verbose: bool = False) -> float:
    """
    Task 2: Win-back a churned high-LTV user within 8 steps, keeping
    notification_fatigue ≤ 3 throughout.

    Scoring (normalized 0.0–1.0):
      Base:           +0.6 if re-engaged
      Early bonus:    +0.2 if re-engaged in ≤ 4 steps
      Spam penalty:   −0.1 per step where fatigue > 3 at time of contact
      No re-engage:   0.0
    """
    env = ReEngageEnv(max_steps=8, seed=seed)
    obs = env.reset()

    # Seek a churned high-LTV user
    attempts = 0
    while not (obs.user_status == "churned" and obs.ltv_bucket == "high") and attempts < 30:
        obs = env.reset()
        attempts += 1

    from reengage_env.environment import CHANNEL_TO_ACTION

    spam_steps    = 0
    re_engaged    = False
    re_engage_step= None

    for step_num in range(1, 9):
        # Agent: wait if fatigued, else use channel pref + discount if sensitive
        if obs.notification_fatigue >= 4:
            action = 5  # wait
        elif obs.channel_pref != "none" and obs.channel_pref in CHANNEL_TO_ACTION:
            action = CHANNEL_TO_ACTION[obs.channel_pref]
        elif obs.discount_sensitivity == "yes":
            action = 3  # discount
        else:
            action = 0  # email

        # Count spam: sending a real action when fatigue > 3
        if action != 5 and obs.notification_fatigue > 3:
            spam_steps += 1

        obs, reward, done, info = env.step(action)
        if verbose:
            print(f"  Step {step_num}: action={action}, fatigue={obs.notification_fatigue}, reward={reward:.3f}")

        if info["re_engaged"] and not re_engaged:
            re_engaged     = True
            re_engage_step = step_num

        if done:
            break

    if not re_engaged:
        score = 0.0
    else:
        score  = 0.6
        if re_engage_step is not None and re_engage_step <= 4:
            score += 0.2
        score -= 0.1 * spam_steps
        score  = max(0.0, min(1.0, round(score, 3)))

    if verbose:
        print(f"Task 2 score: {score:.2f}  (re_engaged={re_engaged}, spam_steps={spam_steps})")
    return score


# ---------------------------------------------------------------------------
# Task 3 — Revenue-aligned multi-user campaign (Hard)
# ---------------------------------------------------------------------------

def grade_task3(seed: int = 0, verbose: bool = False) -> float:
    """
    Task 3: Run a 10-user batch campaign. Maximize revenue proxy
    (LTV-weighted re-engagement rate) with ≤ 5 actions per user.

    Scoring (precision × recall, F1-style, weighted by LTV):
      - precision = high-LTV users re-engaged / total re-engaged
      - recall    = high-LTV users re-engaged / total high-LTV users
      - score     = 2 * precision * recall / (precision + recall + 1e-8)
      Normalized to [0, 1].
    """
    from reengage_env.environment import CHANNEL_TO_ACTION, LTV_MULTIPLIER

    NUM_USERS        = 10
    MAX_STEPS_EACH   = 5

    total_high_ltv  = 0
    re_engaged_high = 0
    re_engaged_any  = 0

    rng_seeds = [seed + i * 17 for i in range(NUM_USERS)]

    for i, s in enumerate(rng_seeds):
        env = ReEngageEnv(max_steps=MAX_STEPS_EACH, seed=s)
        obs = env.reset()

        if obs.ltv_bucket == "high":
            total_high_ltv += 1

        for step_num in range(MAX_STEPS_EACH):
            # Revenue-aligned agent: prioritize high LTV, be conservative with low LTV
            if obs.ltv_bucket == "low":
                # Don't over-invest in low-LTV users
                if step_num == 0:
                    action = 4  # feature_tip (cheap)
                else:
                    action = 5  # wait
            else:
                if obs.notification_fatigue >= 4:
                    action = 5
                elif obs.channel_pref != "none" and obs.channel_pref in CHANNEL_TO_ACTION:
                    action = CHANNEL_TO_ACTION[obs.channel_pref]
                elif obs.discount_sensitivity == "yes" and obs.ltv_bucket == "high":
                    action = 3
                else:
                    action = 0

            obs, reward, done, info = env.step(action)

            if info["re_engaged"]:
                re_engaged_any += 1
                if obs.ltv_bucket == "high" or (
                    obs.ltv_bucket != "high" and env.state().observation.ltv_bucket == "high"
                ):
                    re_engaged_high += 1
                break

            if done:
                break

        if verbose:
            st = env.state()
            print(f"  User {i+1}: ltv={obs.ltv_bucket}, re_engaged={info['re_engaged']}, cumR={st.cumulative_reward:.2f}")

    precision = re_engaged_high / (re_engaged_any + 1e-8)
    recall    = re_engaged_high / (total_high_ltv + 1e-8)
    score     = 2 * precision * recall / (precision + recall + 1e-8)
    score     = min(1.0, max(0.0, round(score, 3)))

    if verbose:
        print(f"Task 3: high_ltv_users={total_high_ltv}, re_engaged_high={re_engaged_high}, "
              f"re_engaged_any={re_engaged_any}, score={score:.3f}")
    return score


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

def grade_all(seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    """
    Run all three graders and return a summary dict.
    Weights: Task1=33%, Task2=33%, Task3=34% (equal for baseline).
    """
    t1 = grade_task1(seed=seed, verbose=verbose)
    t2 = grade_task2(seed=seed, verbose=verbose)
    t3 = grade_task3(seed=seed, verbose=verbose)

    composite = round(0.33 * t1 + 0.33 * t2 + 0.34 * t3, 4)

    result = {
        "task1_score": t1,
        "task2_score": t2,
        "task3_score": t3,
        "composite":   composite,
    }
    if verbose:
        print(f"\n=== Composite Score: {composite:.4f} ===")
    return result
