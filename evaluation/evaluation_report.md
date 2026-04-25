# ReEngageEnv Evaluation Report

## Evaluation Setup

Policy tested: deterministic offline baseline  
Number of episodes: 5  
Seeds: 42, 43, 44, 45, 46  
Environment URL: http://localhost:7860  
Verifier: enabled  
Audit logging: enabled  

## Results Summary

| Metric | Value |
|---|---:|
| Success rate | 1.0 |
| Average reward | 1.668 |
| Average steps | 3.6 |
| Verification pass rate | 1.0 |
| Fatigue violation rate | 0.0 |
| Channel match rate | 0.5833 |
| Discount usage rate | 0.0 |

## Interpretation

The baseline policy successfully re-engaged users across all tested seeds.

The strongest result is safety: the fatigue violation rate is 0.0 and the verification pass rate is 1.0. This means the current rollout policy avoids unsafe repeated-contact behavior under the verifier rules.

The weakest result is channel preference alignment. Channel match rate is 0.5833, meaning the policy still misses user preference in a meaningful number of contact actions.

Discount usage is 0.0. This is safe, but it may be too conservative for high-LTV discount-sensitive users.

## Strengths

- All evaluated episodes succeeded.
- No fatigue violations were detected.
- All trajectories passed verification.
- Average episode length stayed low.
- Logs were generated for reproducibility.

## Weaknesses

- Channel matching needs improvement.
- Discount strategy is underused.
- Current policy is deterministic, not trained.
- Evaluation set is small.
- Results are baseline-only, not baseline-vs-trained yet.

## Next Improvement Targets

1. Improve channel preference matching.
2. Add controlled discount usage for high-LTV discount-sensitive users.
3. Run more seeds.
4. Add trained model comparison.
5. Add GRPO/TRL training loop.

## Current Conclusion

The project now has a working evaluation loop with measurable metrics, verifier checks, and trajectory logs.

This proves the environment and rollout system are operational.

The next major compliance gap is reinforcement learning training.