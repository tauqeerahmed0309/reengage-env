import gradio as gr
import requests

ENV_URL = "http://localhost:7860"

obs = None

def reset_env():
    global obs
    res = requests.post(f"{ENV_URL}/reset").json()
    obs = res
    return res

def step_env(action):
    global obs
    res = requests.post(f"{ENV_URL}/step", json={"action": action}).json()
    obs = res["observation"]
    return res

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 ReEngageEnv Interactive Demo")

    reset_btn = gr.Button("Reset Environment")
    action = gr.Dropdown(
        choices=[
            ("email", 0),
            ("push", 1),
            ("sms", 2),
            ("discount", 3),
            ("feature_tip", 4),
            ("wait", 5),
        ],
        label="Select Action"
    )
    step_btn = gr.Button("Take Step")

    output = gr.JSON()

    reset_btn.click(fn=reset_env, outputs=output)
    step_btn.click(fn=step_env, inputs=action, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)