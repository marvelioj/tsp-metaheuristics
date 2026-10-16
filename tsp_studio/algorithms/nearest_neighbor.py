"""Nearest Neighbor construction."""
from __future__ import annotations
import time
from ..core.problem import TSPProblem
from .base import Solver, Solution


class NearestNeighbor(Solver):
    name = "nearest_neighbor"

    def solve(self, problem: TSPProblem, *, start: int = 0,
              initial_tour=None, **_) -> Solution:
        t0 = time.perf_counter()
        n = problem.n

        if initial_tour is not None:
            problem.validate_tour(initial_tour)
            tour = list(initial_tour)
            unvisited = set(range(n)) - set(tour)
            cur = tour[-1]
            while unvisited:
                nxt = min(unvisited, key=lambda j: problem.dist(cur, j))
                tour.append(nxt)
                unvisited.remove(nxt)
                cur = nxt
        else:
            if not 0 <= start < n:
                raise ValueError(f"start must be in [0,{n})")
            unvisited = set(range(n))
            unvisited.remove(start)
            tour = [start]
            cur = start
            while unvisited:
                nxt = min(unvisited, key=lambda j: problem.dist(cur, j))
                tour.append(nxt)
                unvisited.remove(nxt)
                cur = nxt

        return self._make_solution(problem, tour,
                                   time.perf_counter() - t0, start=start)
