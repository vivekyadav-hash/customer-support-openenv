from dataclasses import dataclass
from typing import Generic, TypeVar

ObservationT = TypeVar("ObservationT")

@dataclass
class StepResult(Generic[ObservationT]):
    observation: ObservationT
    reward: float
    done: bool