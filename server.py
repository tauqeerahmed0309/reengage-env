"""
ReEngageEnv — OpenEnv HTTP Server
Exposes the environment via a REST API compatible with the OpenEnv spec.

Endpoints
---------
POST /reset           → Observation
POST /step            → {observation, reward, done, info}
GET  /state           → EnvState
GET  /health          → {"status": "ok"}
GET  /openenv.yaml    → environment metadata
"""

from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from reengage_env.environment import ReEngageEnv, Observation, EnvState

# ---------------------------------------------------------------------------
# App & global env instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ReEngageEnv",
    description="OpenEnv-compliant re-engagement RL environment",
    version="1.0.0",
)

_env: ReEngageEnv = ReEngageEnv(seed=int(os.environ.get("ENV_SEED", "42")))
_initialized: bool = False


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action: int


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "ReEngageEnv", "version": "1.0.0"}


from fastapi import Body

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = Body(default=None)) -> Observation:
    global _env, _initialized

    if req is None:
        seed = int(os.environ.get("ENV_SEED", "42"))
    else:
        seed = req.seed if req.seed is not None else int(os.environ.get("ENV_SEED", "42"))

    _env = ReEngageEnv(seed=seed)
    obs = _env.reset()
    _initialized = True
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    global _initialized
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")

    if req.action not in range(6):
        raise HTTPException(status_code=422, detail=f"action must be 0–5, got {req.action}")

    try:
        obs, reward, done, info = _env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    if not _initialized:
        raise HTTPException(status_code=400, detail="Call /reset before /state.")
    return _env.state()


@app.get("/openenv.yaml")
def openenv_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if os.path.exists(yaml_path):
        return FileResponse(yaml_path, media_type="text/yaml")
    return JSONResponse({"error": "openenv.yaml not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>🚀 ReEngageEnv API is running</h1>
    <p>Try these endpoints:</p>
    <ul>
        <li>/health</li>
        <li>/reset</li>
        <li>/step</li>
    </ul>
    """
