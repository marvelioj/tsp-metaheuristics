"""Pipeline: chain solvers (construction + improvers)."""
from __future__ import annotations
import time
from typing import Sequence
from ..core.problem import TSPProblem
from .base import Solver, Solution


class Pipeline(Solver):
    """Run a sequence of solvers, feeding each output to the next.

    First solver builds a tour from scratch.
    Subsequent solvers receive `initial_tour=<previous tour>`.
    """

    def __init__(self, solvers: Sequence[Solver], name: str | None = None):
        if not solvers:
            raise ValueError("Pipeline needs at least one solver.")
        self.solvers = list(solvers)
        self.name = name or "+".join(s.name for s in self.solvers)

    def solve(self, problem: TSPProblem, **kwargs) -> Solution:
        t0 = time.perf_counter()
        tour = None
        step_log = []

        for solver in self.solvers:
            if tour is None:
                sol = solver.solve(problem, **kwargs)
            else:
                try:
                    sol = solver.solve(problem, initial_tour=tour, **kwargs)
                except TypeError:
                    sol = solver.solve(problem, **kwargs)
            tour = sol.tour
            step_log.append({
                "solver": solver.name,
                "length": sol.length,
                "time": sol.runtime_sec,
            })

        total = time.perf_counter() - t0
        problem.validate_tour(tour)
        return Solution(
            tour=tour, length=problem.tour_length(tour),
            algorithm=self.name, runtime_sec=total,
            metadata={**sol.metadata, "steps": step_log},
        )
