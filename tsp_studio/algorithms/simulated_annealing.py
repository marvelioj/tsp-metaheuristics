"""Simulated Annealing for the TSP (2-opt neighbourhood, geometric cooling).

Ported from optimization_engine.py and wrapped as a `Solver`, so it can be used
through the registry and chained in a `Pipeline`
(e.g. nearest_neighbor + simulated_annealing).
"""
from __future__ import annotations
import math
import random
import time
from typing import Optional

from ..core.problem import TSPProblem
from .base import Solver, Solution
from . import _moves


class SimulatedAnnealing(Solver):
    name = "simulated_annealing"

    def solve(self, problem: TSPProblem, *,
              initial_tour: Optional[list[int]] = None,
              T_max: float = 1000.0,
              T_min: float = 1e-3,
              alpha: float = 0.99,
              max_iter_per_temp: int = 50,
              seed: Optional[int] = None,
              **_) -> Solution:
        """
        T_max / T_min       start and stop temperature (scale them to the
                            distances of your data)
        alpha               geometric cooling factor, T <- alpha * T, in (0, 1)
        max_iter_per_temp   neighbours tried at each temperature
        seed                makes a run reproducible
        initial_tour        starting tour; random if None

        metadata: initial_tour, history_best, history_current, history_temp
        (one entry per temperature level), insights, params.
        """
        t0 = time.perf_counter()
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if not 0 < T_min < T_max:
            raise ValueError("Need 0 < T_min < T_max.")
        if max_iter_per_temp < 1:
            raise ValueError("max_iter_per_temp must be >= 1.")

        n, mat = problem.n, problem.distance_matrix
        _moves.require_symmetric(mat)
        rng = random.Random(seed)

        if initial_tour is None:
            current = list(range(n))
            rng.shuffle(current)
        else:
            problem.validate_tour(initial_tour)
            current = list(initial_tour)

        start_tour = list(current)
        initial_len = problem.tour_length(current)
        cur_len = best_len = initial_len
        best = list(current)

        T = T_max
        h_best, h_cur, h_temp = [best_len], [cur_len], [T]
        uphill_total = uphill_accepted = 0
        total_iter = conv_iter = 0

        if n >= 4:   # for n == 3 every tour is optimal
            while T > T_min:
                for _i in range(max_iter_per_temp):
                    total_iter += 1
                    i, j = _moves.random_move(rng, n)
                    d = _moves.delta(current, i, j, mat)
                    if d < 0:
                        accept = True
                    else:
                        uphill_total += 1
                        accept = rng.random() < math.exp(-d / T)   # Metropolis
                        if accept:
                            uphill_accepted += 1
                    if accept:
                        _moves.apply(current, i, j)
                        cur_len += d
                        if cur_len < best_len - 1e-12:
                            best_len = cur_len
                            best = list(current)
                            conv_iter = total_iter
                h_best.append(best_len)
                h_cur.append(cur_len)
                h_temp.append(T)
                T *= alpha

        insights = {
            "algorithm": "Simulated Annealing",
            "initial_distance": initial_len,
            "best_distance": best_len,
            "improvement_pct": ((initial_len - best_len) / initial_len * 100
                                if initial_len > 0 else 0.0),
            "total_iterations": total_iter,
            "convergence_iter": conv_iter,
            "accepted_uphill_moves": uphill_accepted,
            "uphill_accept_rate": (uphill_accepted / uphill_total * 100
                                   if uphill_total else 0.0),
            "final_temperature": T,
        }
        return self._make_solution(
            problem, best, time.perf_counter() - t0,
            initial_tour=start_tour,
            history_best=h_best, history_current=h_cur, history_temp=h_temp,
            insights=insights,
            params=dict(T_max=T_max, T_min=T_min, alpha=alpha,
                        max_iter_per_temp=max_iter_per_temp, seed=seed),
        )
