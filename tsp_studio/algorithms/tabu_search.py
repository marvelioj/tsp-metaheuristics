"""Tabu Search for the TSP (2-opt neighbourhood, aspiration by best-so-far).

Ported from optimization_engine.py and wrapped as a `Solver`.

Tabu attribute: the move (i, j) (positions of the reversed segment). A move
chosen at iteration t is tabu for iterations t+1 .. t+tabu_tenure.

Two neighbourhood modes:
  * deterministic=False (default): `neighborhood_size` random moves per iteration
  * deterministic=True: ALL valid 2-opt moves per iteration. Use this for small
    instances and for the n=6 illustration; set record_log=True to get a
    per-iteration table (tour, active tabu list, every candidate, chosen move).
"""
from __future__ import annotations
import random
import time
from typing import Optional

from ..core.problem import TSPProblem
from .base import Solver, Solution
from . import _moves


class TabuSearch(Solver):
    name = "tabu_search"

    def solve(self, problem: TSPProblem, *,
              initial_tour: Optional[list[int]] = None,
              max_iterations: int = 500,
              tabu_tenure: int = 20,
              neighborhood_size: int = 50,
              deterministic: bool = False,
              record_log: bool = False,
              seed: Optional[int] = None,
              **_) -> Solution:
        """
        metadata: initial_tour, history_best, history_current, insights, params,
        and `log` (list of per-iteration dicts) when record_log=True.
        """
        t0 = time.perf_counter()
        if max_iterations < 1 or tabu_tenure < 0 or neighborhood_size < 1:
            raise ValueError("max_iterations, neighborhood_size >= 1 and "
                             "tabu_tenure >= 0 are required.")

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

        tabu: dict[tuple[int, int], int] = {}     # move -> last iteration it is tabu
        h_best, h_cur = [best_len], [cur_len]
        aspiration_hits = blocked = conv_iter = 0
        log: list[dict] = []
        moves_all = _moves.all_moves(n) if deterministic else None

        iterations = max_iterations if n >= 4 else 0   # n == 3: nothing to improve
        for it in range(1, iterations + 1):
            candidates = (moves_all if deterministic else
                          [_moves.random_move(rng, n) for _ in range(neighborhood_size)])

            if record_log:
                tour_before = list(current)
                tabu_before = sorted(
                    (m, exp - it + 1) for m, exp in tabu.items() if exp >= it)
                cand_log = []

            chosen = None
            chosen_delta = float("inf")
            for mv in candidates:
                d = _moves.delta(current, mv[0], mv[1], mat)
                new_len = cur_len + d
                is_tabu = tabu.get(mv, 0) >= it
                aspiration = new_len < best_len - 1e-12
                if is_tabu and aspiration:
                    aspiration_hits += 1
                allowed = (not is_tabu) or aspiration
                if not allowed:
                    blocked += 1
                if record_log:
                    cand_log.append({
                        "move": mv, "length": new_len,
                        "status": ("aspiration" if is_tabu and aspiration
                                   else "tabu" if is_tabu else "allowed")})
                if allowed and d < chosen_delta:
                    chosen, chosen_delta = mv, d

            if chosen is not None:
                _moves.apply(current, chosen[0], chosen[1])
                cur_len += chosen_delta
                tabu[chosen] = it + tabu_tenure
                if cur_len < best_len - 1e-12:
                    best_len = cur_len
                    best = list(current)
                    conv_iter = it

            h_best.append(best_len)
            h_cur.append(cur_len)
            if record_log:
                log.append({
                    "iteration": it, "tour_before": tour_before,
                    "tabu_list": tabu_before,        # [((i, j), iterations_left)]
                    "candidates": cand_log, "chosen": chosen,
                    "tour_after": list(current), "length": cur_len,
                    "best_length": best_len})

        insights = {
            "algorithm": "Tabu Search",
            "initial_distance": initial_len,
            "best_distance": best_len,
            "improvement_pct": ((initial_len - best_len) / initial_len * 100
                                if initial_len > 0 else 0.0),
            "total_iterations": iterations,
            "convergence_iter": conv_iter,
            "aspiration_hits": aspiration_hits,
            "tabu_blocked_moves": blocked,
            "tabu_tenure": tabu_tenure,
        }
        meta = dict(
            initial_tour=start_tour, history_best=h_best, history_current=h_cur,
            insights=insights,
            params=dict(max_iterations=max_iterations, tabu_tenure=tabu_tenure,
                        neighborhood_size=neighborhood_size,
                        deterministic=deterministic, seed=seed))
        if record_log:
            meta["log"] = log
        return self._make_solution(problem, best, time.perf_counter() - t0, **meta)
