"""2-opt local search. Standalone or as post-optimizer."""
from __future__ import annotations
import time
from typing import Optional, Sequence
from ..core.problem import TSPProblem
from .base import Solver, Solution


def two_opt_improve(tour, problem, max_passes=None):
    """In-place 2-opt. Returns (new_tour, length)."""
    n = len(tour)
    if n < 4:
        return list(tour), problem.tour_length(tour)

    tour = list(tour)
    mat = problem.distance_matrix
    improved = True
    passes = 0

    while improved:
        improved = False
        passes += 1
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for k in range(i + 2, n):
                c = tour[k]
                d = tour[(k + 1) % n]
                if d == a:
                    continue
                delta = mat[a][c] + mat[b][d] - mat[a][b] - mat[c][d]
                if delta < -1e-9:
                    tour[i + 1:k + 1] = reversed(tour[i + 1:k + 1])
                    improved = True
                    break
            if improved:
                break
        if max_passes is not None and passes >= max_passes:
            break

    return tour, problem.tour_length(tour)


class TwoOpt(Solver):
    name = "two_opt"

    def solve(self, problem, *, initial_tour=None,
              max_passes=None, **_):
        t0 = time.perf_counter()

        if initial_tour is None:
            from .nearest_neighbor import NearestNeighbor
            initial_tour = NearestNeighbor().solve(problem).tour
        else:
            problem.validate_tour(initial_tour)
            initial_tour = list(initial_tour)

        tour, length = two_opt_improve(initial_tour, problem, max_passes)
        return self._make_solution(problem, tour,
                                   time.perf_counter() - t0,
                                   max_passes=max_passes)
