# ReEngageEnv Hackathon Compliance Plan

## Goal
Build an OpenEnv-compatible reinforcement learning environment for user re-engagement optimization, with verifiable rewards, anti-hacking safeguards, RL training, evaluation, and demo evidence.

## Current Compliance Status

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Step-based environment | Done | reset/step/state API exists | None |
| Objective reward | Partial | shaped reward exists | Needs independent verifier |
| OpenEnv-compatible deployment | Partial | FastAPI + Docker exists | Need final deployed Space proof |
| Baseline agent | Done | inference loop exists | Action space mismatch |
| RL training | Missing | No TRL/GRPO script yet | Need train_grpo.py |
| Anti-reward-hacking checks | Missing | No verifier layer | Need verifier module |
| Evaluation report | Missing | No before/after comparison | Need metrics + report |
| Demo | Missing | No user-facing demo | Need demo app/scenarios |

## Critical Issues

### 1. Action Space Mismatch
Environment defines six actions:
- email
- push
- sms
- discount
- feature_tip
- wait

Current inference baseline only uses:
- email
- notification
- ignore

This must be fixed in a new additive baseline or rollout file, without modifying existing inference.py.

### 2. Missing RL Training
The project currently demonstrates inference, not reinforcement learning training.

Needed:
- rollout collection
- reward extraction
- TRL/GRPO trainer
- model save path
- evaluation before and after training

### 3. Weak Reward-Hacking Defense
The reward function exists, but there is no separate verifier checking:
- invalid action usage
- fatigue abuse
- repeated contact spam
- impossible reward jumps
- suspicious termination behavior

## Additive Implementation Plan

### Phase 1 — Documentation
- [ ] architecture.md
- [ ] reward_design.md
- [ ] evaluation_report.md

### Phase 2 — Verification
- [ ] reward_verifier.py
- [ ] safety_checks.py
- [ ] audit_logger.py

### Phase 3 — Evaluation
- [ ] metrics.py
- [ ] baseline experiment logs
- [ ] trained experiment logs

### Phase 4 — Training
- [ ] rollout.py
- [ ] config.yaml
- [ ] train_grpo.py

### Phase 5 — Demo
- [ ] demo/app.py
- [ ] demo/scenarios.json
- [ ] final screenshots or video

## Target Final Compliance Score

Current estimate: 65–70%

Target after additions: 90%+

## Judge-Facing Story

ReEngageEnv turns user re-engagement into a sequential decision-making problem.  
The agent learns when to contact, which channel to use, and when to wait, while balancing revenue impact against notification fatigue.

The final demo should show:
1. baseline agent behavior,
2. verifier/reward breakdown,
3. trained model behavior,
4. measurable improvement,
5. safety checks against spam and reward hacking.