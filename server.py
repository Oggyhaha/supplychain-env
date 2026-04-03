# ## FILE 6 — `server.py`

# ### What Is It?
# ```
# Makes our environment accessible as a web API.
# Judges ping this to validate our environment.

# Without server.py → only works locally
# With server.py    → works anywhere on internet
#                     judges can test it remotely
# Open server.py and paste:
# python# server.py
# # ═══════════════════════════════════════════════
# # FASTAPI SERVER
# # Exposes environment as REST API endpoints
# # ═══════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from env.environment import SupplyChainEnvironment
from env.models import RestockAction
import uvicorn

app = FastAPI(
    title="SupplyChain-Env",
    description="OpenEnv supply chain management environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store environments per task
environments = {}


@app.get("/")
def root():
    return {
        "name": "SupplyChain-Env",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset/{task_id}")
def reset(task_id: str):
    """Reset environment and start new episode"""
    if task_id not in ["task_easy", "task_medium", "task_hard"]:
        raise HTTPException(status_code=404, detail="Task not found")
    env = SupplyChainEnvironment(task_id)
    result = env.reset()
    environments[task_id] = env
    return result.model_dump()


@app.post("/step/{task_id}")
def step(task_id: str, action: RestockAction):
    """Take one step in the environment"""
    if task_id not in environments:
        raise HTTPException(
            status_code=400,
            detail="Call /reset first"
        )
    env = environments[task_id]
    result = env.step(action)
    return result.model_dump()


@app.get("/state/{task_id}")
def state(task_id: str):
    """Get current environment state"""
    if task_id not in environments:
        raise HTTPException(
            status_code=400,
            detail="Call /reset first"
        )
    return environments[task_id].state().model_dump()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)