# ReEngageEnv Architecture

## System Overview

ReEngageEnv is a reinforcement learning environment for user re-engagement optimization.

The system is split into five layers:

1. Environment
2. Verifier
3. Rollout Collection
4. RL Training
5. Evaluation and Demo

## 1. Environment Layer

The environment simulates user re-engagement scenarios.

Core API:

- `reset(seed)` starts a new episode.
- `step(action)` applies an outreach decision.
- `state()` returns full environment state.
- `health()` confirms the server is running.

The agent receives an observation containing user behavior, monetization value, engagement context, segmentation, and action history.

## 2. Action Layer

The intended action space contains six actions:

| ID | Action | Meaning |
|---|---|---|
| 0 | email | Send email |
| 1 | push | Send push notification |
| 2 | sms | Send SMS |
| 3 | discount | Send promo/discount |
| 4 | feature_tip | Highlight product feature |
| 5 | wait | Do not contact |

## 3. Reward Layer

The reward encourages:

- successful re-engagement,
- matching the user's preferred channel,
- avoiding notification fatigue,
- avoiding wasteful discounts,
- waiting when fatigue is high,
- solving the task efficiently.

The reward is shaped per step instead of only at episode end.

## 4. Verifier Layer

The verifier is separate from the environment.

It checks whether trajectories are valid and safe.

Verifier responsibilities:

- validate actions,
- detect fatigue violations,
- detect repeated spam behavior,
- detect impossible reward jumps,
- check episode termination consistency,
- produce audit logs for suspicious rollouts.

This protects the system from reward hacking.

## 5. Rollout Layer

The rollout layer interacts with the environment server.

It is responsible for:

- resetting episodes,
- sending actions,
- collecting observations,
- collecting rewards,
- storing trajectories,
- preparing data for RL training and evaluation.

This layer must support all six environment actions.

## 6. Training Layer

The training layer uses TRL/GRPO-style reinforcement learning.

Training flow:

1. Model receives observation prompt.
2. Model outputs an action.
3. Action is executed in the environment.
4. Environment returns reward and next observation.
5. Verifier checks the transition.
6. Trainer updates the model toward higher-reward behavior.

Training should start small before scaling.

## 7. Evaluation Layer

Evaluation compares:

- baseline model,
- trained model,
- rule-based sanity policy if needed.

Metrics:

- average reward,
- success rate,
- average episode length,
- fatigue violation rate,
- high-LTV re-engagement rate,
- invalid action rate.

## 8. Demo Layer

The demo should show:

1. user scenario,
2. baseline action sequence,
3. reward breakdown,
4. trained action sequence,
5. improvement summary,
6. safety/verifier output.

## Final Architecture Flow

```text
Observation
    ↓
LLM Policy
    ↓
Action
    ↓
Environment Server
    ↓
Reward + Next Observation
    ↓
Verifier / Safety Checks
    ↓
Trajectory Logs
    ↓
TRL / GRPO Training
    ↓
Evaluation + Demo