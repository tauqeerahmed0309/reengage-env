import os
import requests
from openai import OpenAI

# Environment variables (injected by evaluator)
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# Initialize OpenAI client (MANDATORY)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


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

        # Safety check
        if action not in [0, 1, 2]:
            return 0

        return action

    except Exception as e:
        print(f"LLM error: {e}")
        return 0  # fallback


def run_task(task_id, seed):
    print(f"[START] task={task_id} seed={seed}")

    res = requests.post(f"{ENV_URL}/reset", json={"seed": seed})
    obs = res.json()

    total_reward = 0

    for t in range(10):
        # 🔥 LLM CALL (this is what validator checks)
        action = get_action_from_llm(obs)

        step_res = requests.post(
            f"{ENV_URL}/step",
            json={"action": action}
        ).json()

        reward = step_res["reward"]
        done = step_res["done"]

        # IMPORTANT: update observation
        obs = step_res.get("observation", obs)

        total_reward += reward

        print(f"[STEP] t={t+1} action={action} reward={reward}")

        if done:
            break

    score = min(1.0, max(0.0, total_reward / 5.0))
    print(f"[END] score={score}")

    return score


def main():
    seeds = [42, 43, 44]

    for task_id in [1, 2, 3]:
        for seed in seeds:
            run_task(task_id, seed)


if __name__ == "__main__":
    main()
