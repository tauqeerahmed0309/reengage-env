import os
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def run_task(task_id, seed):
    print(f"[START] task={task_id} seed={seed}")

    res = requests.post(f"{ENV_URL}/reset", json={"seed": seed})
    obs = res.json()

    total_reward = 0

    for t in range(10):
        action = 0  # simple baseline: always email

        step_res = requests.post(
            f"{ENV_URL}/step",
            json={"action": action}
        ).json()

        reward = step_res["reward"]
        done = step_res["done"]

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