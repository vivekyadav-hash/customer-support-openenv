from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CustomerSupportAction:
    priority: str
    department: str
    response_draft: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomerSupportObservation:
    ticket_id: str
    ticket_text: str
    task_level: str
    done: bool
    reward: Optional[float]
    feedback: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomerSupportState:
    episode_id: str
    step_count: int
    task_level: str
    current_ticket: str
    total_reward: float