from __future__ import annotations
from typing import Iterable
from ..core.problem import TSPProblem
from ..algorithms.base import Solution, Solver


def run_solver(solver: Solver, problem: TSPProblem, **kwargs) -> Solution:
    return solver.solve(problem, **kwargs)


def benchmark(solvers: Iterable[Solver], problem: TSPProblem,
              **kwargs) -> list[Solution]:
    results = []
    for s in solvers:
        try:
            results.append(run_solver(s, problem, **kwargs))
        except Exception as e:
            print(f"[benchmark] {s.name} failed: {e}")
    results.sort(key=lambda r: r.length)
    return results


def improvement_pct(baseline: float, improved: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - improved) / baseline * 100.0
