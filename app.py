import gradio as gr
import requests

ENV_URL = "http://localhost:7860"

def reset_env():
    return requests.post(f"{ENV_URL}/reset").json()

def step_env(action):
    return requests.post(f"{ENV_URL}/step", json={"action": action}).json()

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 ReEngageEnv Interactive Demo")

    reset_btn = gr.Button("Reset Environment")

    action = gr.Radio(
        choices=[
            ("Email", 0),
            ("Push", 1),
            ("SMS", 2),
            ("Discount", 3),
            ("Feature Tip", 4),
            ("Wait", 5),
        ],
        label="Choose Action"
    )

    step_btn = gr.Button("Take Step")

    output = gr.JSON()

    reset_btn.click(fn=reset_env, outputs=output)
    step_btn.click(fn=step_env, inputs=action, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)