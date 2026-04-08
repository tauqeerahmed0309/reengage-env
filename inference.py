import os
import requests
from openai import OpenAI

# Environment variables injected by evaluator
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Initialize OpenAI client (MANDATORY)
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

MAX_STEPS = 10

def get_action_from_llm(obs):
    prompt = f"""
You are an intelligent agent in a user re-engagement system.

Observation:
{obs}

Choose the best action:
0 = send email
1 = send notification
2 = ignore

Return ONLY a number (0, 1, or 2).
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        action_text = response.choices[0].message.content.strip()
        action = int(action_text)
        if action not in [0, 1, 2]:
            return 0
        return action
    except Exception as e:
        print(f"LLM error: {e}")
        return 0  # fallback


def run_task(task_id, seed):
    print(f"[START] task={task_id} env=reengage-env model={MODEL_NAME}")

    res = requests.post(f"{ENV_URL}/reset", json={"seed": seed})
    obs = res.json()

    total_reward = 0.0
    rewards = []

    for t in range(1, MAX_STEPS + 1):
        action = get_action_from_llm(obs)

        step_res = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
        reward = round(step_res.get("reward", 0.0), 2)
        done = step_res.get("done", False)
        error = step_res.get("error", None)

        obs = step_res.get("observation", obs)

        total_reward += reward
        rewards.append(f"{reward:.2f}")

        print(f"[STEP] step={t} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}")

        if done:
            break

    score = min(1.0, max(0.0, total_reward / 5.0))
    print(f"[END] success={str(score>0).lower()} steps={len(rewards)} score={score:.2f} rewards={','.join(rewards)}")
    return score


def main():
    seeds = [42, 43, 44]
    for task_id in [1, 2, 3]:
        for seed in seeds:
            run_task(task_id, seed)


if __name__ == "__main__":
    main()
