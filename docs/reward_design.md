# ReEngageEnv Reward Design

## Reward Goal

The reward function should teach the agent to re-engage users without spamming them or wasting budget.

The agent should learn:

- who to contact,
- which channel to use,
- when to wait,
- when a discount is justified,
- how to avoid notification fatigue.

## Reward Components

The shaped reward is composed of:

```text
reward =
    re_engagement_reward
  + channel_match_bonus
  + fatigue_penalty
  + discount_penalty
  + wait_bonus
  + step_cost