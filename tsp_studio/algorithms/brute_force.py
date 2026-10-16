"""Exact TSP via Held–Karp DP (feasible for n ≤ ~18)."""
from __future__ import annotations
import time
from ..core.problem import TSPProblem
from .base import Solver, Solution


class HeldKarp(Solver):
    name = "held_karp_exact"
    MAX_N = 18

    def solve(self, problem: TSPProblem, **_) -> Solution:
        t0 = time.perf_counter()
        n = problem.n
        if n > self.MAX_N:
            raise ValueError(
                f"Held-Karp limited to n <= {self.MAX_N}; got {n}. "
                "Use a heuristic instead."
            )
        mat = problem.distance_matrix
        INF = float("inf")

        # dp[mask][j] = min cost path visiting `mask` starting at 0, ending at j
        size = 1 << n
        dp = [[INF] * n for _ in range(size)]
        parent = [[-1] * n for _ in range(size)]
        dp[1][0] = 0.0

        for mask in range(size):
            if not (mask & 1):
                continue
            for j in range(n):
                if not (mask & (1 << j)) or dp[mask][j] == INF:
                    continue
                for k in range(1, n):
                    if mask & (1 << k):
                        continue
                    nmask = mask | (1 << k)
                    cand = dp[mask][j] + mat[j][k]
                    if cand < dp[nmask][k]:
                        dp[nmask][k] = cand
                        parent[nmask][k] = j

        full = size - 1
        best_len = INF
        best_last = -1
        for j in range(1, n):
            cand = dp[full][j] + mat[j][0]
            if cand < best_len:
                best_len = cand
                best_last = j

        # reconstruct
        tour = []
        mask = full
        j = best_last
        while j != -1:
            tour.append(j)
            pj = parent[mask][j]
            mask ^= (1 << j)
            j = pj
        # the loop above already ends at city 0, so no extra append
        tour.reverse()

        return self._make_solution(problem, tour, time.perf_counter() - t0)
