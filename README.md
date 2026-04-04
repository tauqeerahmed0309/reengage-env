ReEngageEnv
OpenEnv-compliant RL environment for user re-engagement optimization.
Train and evaluate agents that learn when and how to contact dormant or churned users across email, push, SMS, and discount channels — while respecting notification fatigue and maximizing revenue-aligned re-engagement.
---
Motivation
Re-engaging lapsed users is one of the highest-ROI activities in product growth, yet most systems rely on static rule-based logic (e.g. "send email after 7 days of inactivity"). This environment frames re-engagement as a sequential decision problem, enabling RL agents to learn nuanced, context-aware policies that:
Match outreach channel to user preference
Balance recency signals with monetization value (LTV)
Avoid notification fatigue that drives unsubscribes
Adapt timing to behavioral cluster and discount sensitivity
---
Environment Description
Property	Value
Action space	`Discrete(6)` — email, push, sms, discount, feature_tip, wait
Observation space	20-field typed Pydantic model (see below)
Reward range	`[-1.0, 2.0]` shaped per step
Episode length	`max_steps` (default 10)
Termination	User re-engaged OR max steps reached
Reproducible	Yes — deterministic given same seed
---
Action Space
ID	Name	Description
0	email	Send a personalized email
1	push	Send a mobile push notification
2	sms	Send an SMS text message
3	discount	Send a discount/promo code
4	feature_tip	Highlight a new or underused feature
5	wait	No contact this step; fatigue decays
---
Observation Space
User Behavior Features
Field	Type	Range / Values	Description
`days_since_active`	int	[0, 365]	Days since last session
`session_freq_7d`	int	[0, 7]	Sessions in last 7 days
`session_freq_30d`	int	[0, 30]	Sessions in last 30 days
`avg_session_dur`	float	[0.0, 120.0] min	Average session duration
`features_used`	int	[0, 10]	Distinct product features touched
`drop_off_point`	enum	onboarding/dashboard/checkout/settings/other	Last stage before drop-off
Value / Monetization Features
Field	Type	Range / Values	Description
`total_spend`	float	[0.0, 10000.0] USD	Cumulative lifetime spend
`avg_order_value`	float	[0.0, 500.0] USD	Average transaction value
`subscription_status`	enum	free/trial/paid/cancelled	Current subscription tier
`ltv_bucket`	enum	low/medium/high	LTV segment; drives reward multiplier
Engagement Context
Field	Type	Range / Values	Description
`time_of_day`	int	[0, 23]	Current hour UTC
`day_of_week`	int	[0, 6] Mon=0	Day of week
`device_type`	enum	mobile/desktop/tablet	Primary device
`notification_fatigue`	int	[0, 10]	Messages sent this episode
`channel_pref`	enum	email/push/sms/none	User's preferred contact channel
User Segmentation
Field	Type	Values	Description
`user_status`	enum	new/active/dormant/churned	Activity status
`behavioral_cluster`	enum	browser/buyer/lurker/power_user	Behavioral archetype
`discount_sensitivity`	enum	yes/no/unknown	Discount responsiveness
Action History
Field	Type	Range / Values	Description
`last_action`	enum	all action names	Previous agent action
`num_attempts`	int	[0, 50]	Total outreach attempts this episode
`days_since_last_notif`	int	[0, 365]	Days since most recent notification
---
Reward Function
The reward is shaped per step to provide dense learning signal:
```
reward = re_engagement_reward
       + channel_match_bonus
       + fatigue_penalty
       + discount_penalty
       + wait_bonus
       + step_cost
```
Component	Value
Re-engagement reward	`+1.0 × ltv_multiplier` (low=1.0, medium=1.5, high=2.0)
Channel-match bonus	`+0.2` if action matches `channel_pref`
Fatigue penalty	`−0.15 × (fatigue − 3)` when fatigue ≥ 4
Discount penalty	`−0.10` if discount sent to low-LTV user
Wait bonus	`+0.05` if agent waits when fatigue ≥ 3
Step cost	`−0.02` per step (efficiency pressure)
---
Tasks
Task 1 — Re-engage a dormant user (Easy)
Re-engage a user who has been inactive 7–14 days within 5 steps.
Score 1.0: re-engaged in ≤ 3 steps
Score 0.5: re-engaged in ≤ 5 steps
Score 0.0: not re-engaged
Task 2 — Win-back a high-value churned user (Medium)
Re-engage a churned high-LTV user within 8 steps without triggering fatigue (keep `notification_fatigue ≤ 3`).
Base: +0.6 if re-engaged
Early bonus: +0.2 if re-engaged in ≤ 4 steps
Spam penalty: −0.1 per step where fatigue > 3 at time of contact
Task 3 — Revenue-aligned multi-user campaign (Hard)
Manage a batch of 10 diverse users with ≤ 5 actions each. Maximize LTV-weighted re-engagement.
Scored on precision × recall F1 for high-LTV users
Low-LTV users should be treated conservatively to save budget
---
API Reference
`POST /reset`
```json
{ "seed": 42 }
```
Returns: `Observation` object.
`POST /step`
```json
{ "action": 0 }
```
Returns:
```json
{
  "observation": { ... },
  "reward": 0.18,
  "done": false,
  "info": {
    "re_engaged": false,
    "step": 1,
    "cumulative_reward": 0.18,
    "reward_breakdown": { ... }
  }
}
```
`GET /state`
Returns full `EnvState` including hidden `churn_probability` (for graders only).
`GET /health`
Returns `{"status": "ok"}`.
---
Setup & Installation
Local (Python)
```bash
git clone https://github.com/your-org/reengage-env
cd reengage-env
pip install -e ".[dev]"

# Start the server
python server.py

# Run tests
pytest tests/ -v

# Run baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python scripts/baseline_inference.py
```
Docker
```bash
docker build -t reengage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... reengage-env
```
Hugging Face Spaces
Push to a HF Space repo tagged `openenv`. The server starts automatically on port 7860.
---
Python Usage
```python
from reengage_env import ReEngageEnv

env = ReEngageEnv(seed=42)
obs = env.reset()

print(obs.user_status)          # "dormant"
print(obs.channel_pref)         # "email"
print(obs.notification_fatigue) # 2

# Send email (action 0)
obs, reward, done, info = env.step(0)
print(reward)                   # e.g. 0.18
print(info["re_engaged"])       # False / True

# Full grading
from reengage_env import grade_all
results = grade_all(seed=42, verbose=True)
print(results)
# {"task1_score": 1.0, "task2_score": 0.7, "task3_score": 0.6, "composite": 0.765}
```
---
Project Structure
```
reengage-env/
├── reengage_env/
│   ├── __init__.py          # Public API
│   ├── environment.py       # Core OpenEnv environment
│   └── graders.py           # Task graders (deterministic)
├── tests/
│   └── test_environment.py  # pytest test suite
├── scripts/
│   └── baseline_inference.py# LLM agent baseline runner
├── server.py                # FastAPI HTTP server
├── openenv.yaml             # Environment metadata
├── Dockerfile               # Container definition
├── requirements.txt
├── pyproject.toml
└── README.md
```
---
Environment Variables
Variable	Default	Description
`OPENAI_API_KEY`	—	Required for baseline inference
`OPENAI_BASE_URL`	`https://api.openai.com/v1`	Override for other providers
`ENV_URL`	`http://localhost:7860`	Server URL for baseline script
`MODEL`	`gpt-4o-mini`	LLM model for baseline agent
`ENV_SEED`	`42`	Default RNG seed for the server
`PORT`	`7860`	Server listen port
`NUM_EPISODES`	`3`	Episodes per task in baseline run
---
License
MIT