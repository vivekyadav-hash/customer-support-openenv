import uuid
from typing import Optional
from envs.customer_support_env.models import (
    CustomerSupportAction,
    CustomerSupportObservation,
    CustomerSupportState,
)

TICKETS = {
    "easy": {
        "ticket_id": "TKT-001",
        "ticket_text": "Hi, I was charged twice for my subscription this month. Please refund the extra charge as soon as possible. My account email is john.doe@example.com.",
        "correct_priority": "high",
        "correct_department": "billing",
        "keywords": ["refund", "charge", "billing", "subscription", "account"],
    },
    "medium": {
        "ticket_id": "TKT-002",
        "ticket_text": "Hello, I've been trying to log in for two days but keep getting 'Invalid credentials' even after resetting my password three times. Also, I noticed my last invoice shows a plan I never upgraded to — I'm on the Basic plan but the invoice says Pro. Please sort both issues.",
        "correct_priority": "high",
        "correct_department": "technical",
        "keywords": ["login", "password", "credentials", "invoice", "plan", "reset"],
    },
    "hard": {
        "ticket_id": "TKT-003",
        "ticket_text": "This is absolutely unacceptable! Our entire team of 50 people has been locked out of the platform since 6 AM. We have a critical client demo in 3 hours and your system is DOWN. On top of that, we received an invoice for $4,800 which is 3x our agreed price. I need someone to call me immediately: +1-555-0199. If this isn't resolved in 1 hour I'm cancelling our enterprise contract and filing a chargeback.",
        "correct_priority": "critical",
        "correct_department": "technical",
        "keywords": ["locked out", "down", "demo", "invoice", "chargeback", "enterprise", "cancel", "immediately", "urgent", "outage"],
    },
}


def _grade_response(response_draft: str, keywords: list) -> float:
    score = 0.0
    text = response_draft.lower()
    words = text.split()

    if len(words) >= 20:
        score += 0.10
    elif len(words) >= 10:
        score += 0.05

    hits = sum(1 for kw in keywords if kw.lower() in text)
    coverage = hits / max(len(keywords), 1)
    score += round(coverage * 0.10, 4)

    empathy_words = ["sorry", "apologize", "apologies", "understand", "inconvenience"]
    if any(w in text for w in empathy_words):
        score += 0.05

    action_words = ["will", "team", "resolve", "fix", "escalate", "refund",
                    "contact", "call", "send", "process", "investigate"]
    if any(w in text for w in action_words):
        score += 0.05

    return round(min(score, 0.30), 4)


def grade_action(action: CustomerSupportAction, ticket: dict) -> dict:
    priority_score = 0.40 if action.priority.lower() == ticket["correct_priority"] else 0.0
    dept_score = 0.30 if action.department.lower() == ticket["correct_department"] else 0.0
    response_score = _grade_response(action.response_draft, ticket["keywords"])
    total = round(priority_score + dept_score + response_score, 4)
    return {
        "total_reward": total,
        "priority_score": priority_score,
        "department_score": dept_score,
        "response_score": response_score,
        "correct_priority": ticket["correct_priority"],
        "correct_department": ticket["correct_department"],
    }


class CustomerSupportEnvironment:
    def __init__(self):
        self._episode_id: Optional[str] = None
        self._task_level: Optional[str] = None
        self._ticket: Optional[dict] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False

    def reset(self, task_level: str = "easy") -> CustomerSupportObservation:
        if task_level not in TICKETS:
            raise ValueError(f"task_level must be one of {list(TICKETS.keys())}")
        self._episode_id = str(uuid.uuid4())
        self._task_level = task_level
        self._ticket = TICKETS[task_level]
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        return CustomerSupportObservation(
            ticket_id=self._ticket["ticket_id"],
            ticket_text=self._ticket["ticket_text"],
            task_level=self._task_level,
            done=False,
            reward=None,
            feedback={},
            metadata={"hint": "Assign priority, department, and write a response draft."},
        )

    def step(self, action: CustomerSupportAction) -> CustomerSupportObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        feedback = grade_action(action, self._ticket)
        reward = feedback["total_reward"]
        self._step_count += 1
        self._total_reward += reward
        self._done = True
        return CustomerSupportObservation(
            ticket_id=self._ticket["ticket_id"],
            ticket_text=self._ticket["ticket_text"],
            task_level=self._task_level,
            done=True,
            reward=reward,
            feedback=feedback,
            metadata={},
        )

    @property
    def state(self) -> CustomerSupportState:
        return CustomerSupportState(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            task_level=self._task_level or "",
            current_ticket=self._ticket["ticket_id"] if self._ticket else "",
            total_reward=self._total_reward,
        )