"""
Baseline Inference Script — ReEngageEnv

Runs a language-model agent against the environment using the OpenAI API client.
Reads credentials from environment variables.

Usage
-----
    export OPENAI_API_KEY=sk-...
    export ENV_URL=http://localhost:7860        # or HF Space URL
    python scripts/baseline_inference.py

Environment variables
---------------------
    OPENAI_API_KEY  : required — OpenAI (or compatible) API key
    OPENAI_BASE_URL : optional — override API base (e.g. for Together/Groq)
    ENV_URL         : URL of the running ReEngageEnv server
    MODEL           : model name (default: gpt-4o-mini)
    NUM_EPISODES    : episodes to run per task (default: 3)
    SEED            : RNG seed (default: 42)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config (clean version, no OpenAI required)
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = ""  # not used
OPENAI_BASE_URL = "https://api.openai.com/v1"

ENV_URL         = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL           = "rule-based-agent"
NUM_EPISODES    = int(os.environ.get("NUM_EPISODES", "3"))
SEED            = int(os.environ.get("SEED", "42"))
#if not OPENAI_API_KEY:
    #print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
    #sys.exit(1)

#client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(seed: int | None = None) -> Dict[str, Any]:
    payload = {"seed": seed} if seed is not None else {}
    r = requests.post(f"{ENV_URL}/reset", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def env_step(action: int) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=10)
    r.raise_for_status()
    return r.json()


def env_state() -> Dict[str, Any]:
    r = requests.get(f"{ENV_URL}/state", timeout=10)
    r.raise_for_status()
    return r.json()


def env_health() -> bool:
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "email":       0,
    "push":        1,
    "sms":         2,
    "discount":    3,
    "feature_tip": 4,
    "wait":        5,
}

SYSTEM_PROMPT = """You are a re-engagement agent. Your job is to choose the best action
to re-engage a dormant or churned user. 

Available actions (respond with ONLY the action name, nothing else):
  email       - Send a personalized email
  push        - Send a mobile push notification  
  sms         - Send an SMS text message
  discount    - Send a discount/promo code
  feature_tip - Highlight a new product feature
  wait        - Do nothing this step (reduces fatigue)

Decision guidelines:
- Match the user's channel_pref whenever possible
- Use 'wait' if notification_fatigue >= 4 to avoid spam
- Use 'discount' only for high-value users with discount_sensitivity=yes
- Use 'feature_tip' for power_user behavioral clusters
- Consider days_since_active: recently lapsed users are easier to recover
- Minimize num_attempts for low LTV users (low ROI)

Respond with exactly one of: email, push, sms, discount, feature_tip, wait"""


def llm_choose_action(obs: Dict[str, Any], step_num: int) -> int:
    """
    Rule-based fallback agent (no LLM required)
    """

    # If fatigue high → wait
    if obs["notification_fatigue"] >= 4:
        return 5  # wait

    # Match preferred channel if available
    if obs["channel_pref"] == "email":
        return 0
    elif obs["channel_pref"] == "push":
        return 1
    elif obs["channel_pref"] == "sms":
        return 2

    # High-value + discount sensitive → use discount
    if obs["ltv_bucket"] == "high" and obs["discount_sensitivity"] == "yes":
        return 3

    # Power users → feature tip
    if obs["behavioral_cluster"] == "power_user":
        return 4

    # Default fallback
    return 0  # email

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(episode_seed: int, max_steps: int = 10, verbose: bool = True) -> Dict[str, Any]:
    obs       = env_reset(seed=episode_seed)
    done      = False
    step_num  = 0
    total_rew = 0.0
    re_engaged= False
    history: List[Dict] = []

    if verbose:
        print(f"\n  User: status={obs['user_status']}, ltv={obs['ltv_bucket']}, "
              f"chan={obs['channel_pref']}, fatigue={obs['notification_fatigue']}")

    while not done and step_num < max_steps:
        action    = llm_choose_action(obs, step_num + 1)
        action_name = list(ACTION_MAP.keys())[action]
        result    = env_step(action)
        obs       = result["observation"]
        reward    = result["reward"]
        done      = result["done"]
        info      = result["info"]

        total_rew += reward
        step_num  += 1

        if info.get("re_engaged"):
            re_engaged = True

        history.append({
            "step":        step_num,
            "action":      action_name,
            "reward":      round(reward, 4),
            "re_engaged":  info.get("re_engaged", False),
            "fatigue":     obs["notification_fatigue"],
        })

        if verbose:
            mark = " ✓ RE-ENGAGED" if info.get("re_engaged") else ""
            print(f"  t={step_num:02d}  {action_name:<12}  r={reward:+.3f}  fatigue={obs['notification_fatigue']}{mark}")

        time.sleep(0.1)  # rate-limit courtesy

    return {
        "re_engaged":       re_engaged,
        "total_reward":     round(total_rew, 4),
        "steps":            step_num,
        "history":          history,
    }


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_task(task_num: int, episodes: int, max_steps: int) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Task {task_num} — {episodes} episodes × max {max_steps} steps")
    print('='*60)

    scores:     List[float] = []
    re_engaged: List[bool]  = []

    for ep in range(episodes):
        seed = SEED + task_num * 1000 + ep * 17
        print(f"\n  Episode {ep+1}/{episodes}  (seed={seed})")
        result = run_episode(episode_seed=seed, max_steps=max_steps)
        scores.append(result["total_reward"])
        re_engaged.append(result["re_engaged"])
        print(f"  → total_reward={result['total_reward']:.4f}  re_engaged={result['re_engaged']}  steps={result['steps']}")

    avg_score    = round(sum(scores) / len(scores), 4)
    re_eng_rate  = round(sum(re_engaged) / len(re_engaged), 3)

    print(f"\n  Task {task_num} summary: avg_reward={avg_score:.4f}  re_engagement_rate={re_eng_rate:.1%}")
    return {"task": task_num, "avg_reward": avg_score, "re_engagement_rate": re_eng_rate, "raw_scores": scores}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("ReEngageEnv — Baseline Inference Script")
    print(f"Model:  {MODEL}")
    print(f"EnvURL: {ENV_URL}")

    if not env_health():
        print(f"\nERROR: Cannot reach environment at {ENV_URL}. Is the server running?", file=sys.stderr)
        sys.exit(1)
    print("Environment health: OK\n")

    results = []

    # Task 1: easy, 5 steps
    results.append(run_task(task_num=1, episodes=NUM_EPISODES, max_steps=5))

    # Task 2: medium, 8 steps
    results.append(run_task(task_num=2, episodes=NUM_EPISODES, max_steps=8))

    # Task 3: hard, 10 steps
    results.append(run_task(task_num=3, episodes=NUM_EPISODES, max_steps=10))

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print('='*60)
    for r in results:
        print(f"  Task {r['task']}: avg_reward={r['avg_reward']:.4f}  re_eng_rate={r['re_engagement_rate']:.1%}")

    out_path = "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
