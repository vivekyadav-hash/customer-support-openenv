import requests
from envs.customer_support_env.models import (
    CustomerSupportAction,
    CustomerSupportObservation,
    CustomerSupportState,
)

class CustomerSupportEnv:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_level="easy"):
        resp = requests.post(f"{self.base_url}/reset", json={"task_level": task_level})
        resp.raise_for_status()
        return self._parse(resp.json())

    def step(self, action: CustomerSupportAction):
        resp = requests.post(f"{self.base_url}/step", json={
            "priority": action.priority,
            "department": action.department,
            "response_draft": action.response_draft,
        })
        resp.raise_for_status()
        return self._parse(resp.json())

    def state(self):
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        p = resp.json()
        return CustomerSupportState(**p)

    def _parse(self, p):
        return CustomerSupportObservation(
            ticket_id=p["ticket_id"],
            ticket_text=p["ticket_text"],
            task_level=p["task_level"],
            done=p["done"],
            reward=p.get("reward"),
            feedback=p.get("feedback", {}),
            metadata=p.get("metadata", {}),
        )