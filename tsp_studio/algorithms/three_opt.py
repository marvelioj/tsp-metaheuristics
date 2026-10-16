"""3-opt local search.

Strategy: run 2-opt to convergence, then try 3-opt moves that are verified
to reduce tour length. This guarantees the result is never worse than 2-opt.
"""
from __future__ import annotations
import time
from typing import Optional, Sequence
from ..core.problem import TSPProblem
from .base import Solver, Solution
from .two_opt import two_opt_improve


MAX_PASSES_DEFAULT = 20


def _tour_length(mat, tour):
    n = len(tour)
    return sum(mat[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _try_3opt_move(tour, i, j, k, mat, current_len):
    """Try all 3-opt reconnections for indices (i,j,k). Return (tour, len) or None."""
    n = len(tour)

    # Segment layout (closed tour):
    #   tour = [ ... , A=tour[i], B=tour[i+1], ... , C=tour[j], D=tour[j+1], ... ,
    #            E=tour[k], F=tour[k+1], ... ]
    #
    # Original edges: A-B, C-D, E-F
    # Standard 3-opt reconnections (7 patterns). We test each by constructing
    # the candidate tour explicitly and measuring its length. If shorter than
    # current_len, we accept the best one.
    A, B = tour[i], tour[i + 1]
    C, D = tour[j], tour[j + 1]
    E, F = tour[k], tour[(k + 1) % n]

    seg1 = tour[i + 1:j + 1]     # B..C
    seg2 = tour[j + 1:k + 1]     # D..E
    head = tour[:i + 1]          # ..A
    tail = tour[k + 1:]          # F..

    candidates = [
        seg1[::-1] + seg2,           # reverse seg1
        seg1 + seg2[::-1],           # reverse seg2
        seg1[::-1] + seg2[::-1],     # reverse both
        seg2 + seg1,                 # swap
        seg2 + seg1[::-1],           # swap + reverse seg1
        seg2[::-1] + seg1,           # swap + reverse seg2
        seg2[::-1] + seg1[::-1],     # swap + reverse both
    ]

    best = None
    best_len = current_len
    for new_mid in candidates:
        new_tour = head + new_mid + tail
        if len(new_tour) != n:
            continue
        L = _tour_length(mat, new_tour)
        if L < best_len - 1e-9:
            best_len = L
            best = new_tour

    if best is not None:
        return best, best_len
    return None


def three_opt_improve(tour, problem, max_passes=None):
    n = len(tour)
    if n < 6:
        return two_opt_improve(tour, problem, max_passes)

    if max_passes is None:
        max_passes = MAX_PASSES_DEFAULT

    mat = problem.distance_matrix

    # Step 1: run 2-opt first to get a local optimum
    tour, length = two_opt_improve(list(tour), problem, max_passes)

    # Step 2: try 3-opt moves
    for _pass in range(max_passes):
        improved = False
        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    # avoid wrap-around collision
                    if i == 0 and k == n - 1:
                        continue
                    result = _try_3opt_move(tour, i, j, k, mat, length)
                    if result is not None:
                        tour, length = result
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return tour, length


class ThreeOpt(Solver):
    name = "three_opt"

    def solve(self, problem, *, initial_tour=None, max_passes=None, **_):
        t0 = time.perf_counter()
        if initial_tour is None:
            from .nearest_neighbor import NearestNeighbor
            initial_tour = NearestNeighbor().solve(problem).tour
        else:
            problem.validate_tour(initial_tour)
            initial_tour = list(initial_tour)

        tour, length = three_opt_improve(initial_tour, problem, max_passes)
        return self._make_solution(problem, tour,
                                   time.perf_counter() - t0,
                                   max_passes=max_passes)
