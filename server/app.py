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

@app.get("/metadata")
def metadata():
    return {
        "name": "SupplyChain-Env",
        "description": "OpenEnv supply chain management environment for reinforcement learning",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"]
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer"}
            },
            "required": ["product_id", "quantity"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "inventory": {"type": "object"},
                "sales": {"type": "object"},
                "day": {"type": "integer"},
                "reward": {"type": "number"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "current_inventory": {"type": "object"},
                "total_sales": {"type": "integer"},
                "episode_day": {"type": "integer"}
            }
        }
    }

@app.post("/mcp")
def mcp(request: dict):
    """MCP endpoint for JSON-RPC"""
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "name": "SupplyChain-Env",
            "version": "1.0.0"
        }
    }

# Add this above your existing reset function
@app.post("/reset")
def reset_default():
    """Default reset for OpenEnv Validator"""
    task_id = "task_easy"
    env = SupplyChainEnvironment(task_id)
    result = env.reset()
    environments[task_id] = env
    return result.model_dump()

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


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()