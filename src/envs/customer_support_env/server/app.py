from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from envs.customer_support_env.server.environment import CustomerSupportEnvironment
from envs.customer_support_env.models import CustomerSupportAction

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description="Real-world RL environment: triage customer support tickets.",
    version="1.0.0",
)

env = CustomerSupportEnvironment()


class ResetRequest(BaseModel):
    task_level: str = "easy"


class StepRequest(BaseModel):
    priority: str
    department: str
    response_draft: str
    metadata: Dict[str, Any] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        obs = env.reset(task_level=req.task_level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "ticket_id":   obs.ticket_id,
        "ticket_text": obs.ticket_text,
        "task_level":  obs.task_level,
        "done":        obs.done,
        "reward":      obs.reward,
        "feedback":    obs.feedback,
        "metadata":    obs.metadata,
    }


@app.post("/step")
def step(req: StepRequest):
    action = CustomerSupportAction(
        priority=req.priority,
        department=req.department,
        response_draft=req.response_draft,
        metadata=req.metadata,
    )
    try:
        obs = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "ticket_id":   obs.ticket_id,
        "ticket_text": obs.ticket_text,
        "task_level":  obs.task_level,
        "done":        obs.done,
        "reward":      obs.reward,
        "feedback":    obs.feedback,
        "metadata":    obs.metadata,
    }


@app.get("/state")
def state():
    s = env.state
    return {
        "episode_id":     s.episode_id,
        "step_count":     s.step_count,
        "task_level":     s.task_level,
        "current_ticket": s.current_ticket,
        "total_reward":   s.total_reward,
    }