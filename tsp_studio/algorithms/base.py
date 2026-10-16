from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from ..core.problem import TSPProblem


@dataclass
class Solution:
    tour: list[int]
    length: float
    algorithm: str
    runtime_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Solution(algorithm={self.algorithm!r}, length={self.length:.4f}, "
                f"n={len(self.tour)}, time={self.runtime_sec:.4f}s)")


class Solver(ABC):
    name: str = "base"

    @abstractmethod
    def solve(self, problem: TSPProblem, **kwargs) -> Solution:
        ...

    def _make_solution(self, problem: TSPProblem, tour: list[int],
                       runtime_sec: float, **meta) -> Solution:
        problem.validate_tour(tour)
        length = problem.tour_length(tour)
        return Solution(tour=tour, length=length, algorithm=self.name,
                        runtime_sec=runtime_sec, metadata=meta)
