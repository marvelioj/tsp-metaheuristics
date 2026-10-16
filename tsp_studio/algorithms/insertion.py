"""Insertion heuristics: nearest, cheapest, farthest, arbitrary."""
from __future__ import annotations
import time
from ..core.problem import TSPProblem
from .base import Solver, Solution


class _InsertionBase(Solver):
    """Common machinery for insertion heuristics."""

    def _initial_pair(self, problem, start):
        n = problem.n
        nearest = min((j for j in range(n) if j != start),
                      key=lambda j: problem.dist(start, j))
        return [start, nearest]

    def _best_insertion(self, problem, tour, candidates):
        """Return (city, position, delta) with minimum delta."""
        best = None
        m = len(tour)
        for c in candidates:
            for i in range(m):
                a, b = tour[i], tour[(i + 1) % m]
                delta = problem.dist(a, c) + problem.dist(c, b) - problem.dist(a, b)
                if best is None or delta < best[2]:
                    best = (c, i + 1, delta)
        return best

    def solve(self, problem, *, start=0, initial_tour=None, **_):
        t0 = time.perf_counter()
        n = problem.n

        if initial_tour is not None:
            problem.validate_tour(initial_tour)
            tour = list(initial_tour)
            remaining = set(range(n)) - set(tour)
        else:
            tour = self._initial_pair(problem, start)
            remaining = set(range(n)) - set(tour)

        while remaining:
            self._extend(problem, tour, remaining)

        return self._make_solution(problem, tour,
                                   time.perf_counter() - t0, start=start)

    def _extend(self, problem, tour, remaining):
        raise NotImplementedError


class NearestInsertion(_InsertionBase):
    name = "nearest_insertion"

    def _extend(self, problem, tour, remaining):
        best_city, best_d = None, float("inf")
        for c in remaining:
            d = min(problem.dist(c, t) for t in tour)
            if d < best_d:
                best_d, best_city = d, c
        _, pos, _ = self._best_insertion(problem, tour, [best_city])
        tour.insert(pos, best_city)
        remaining.remove(best_city)


class FarthestInsertion(_InsertionBase):
    name = "farthest_insertion"

    def _extend(self, problem, tour, remaining):
        best_city, best_d = None, -1.0
        for c in remaining:
            d = min(problem.dist(c, t) for t in tour)
            if d > best_d:
                best_d, best_city = d, c
        _, pos, _ = self._best_insertion(problem, tour, [best_city])
        tour.insert(pos, best_city)
        remaining.remove(best_city)


class CheapestInsertion(_InsertionBase):
    name = "cheapest_insertion"

    def _extend(self, problem, tour, remaining):
        city, pos, _ = self._best_insertion(problem, tour, remaining)
        tour.insert(pos, city)
        remaining.remove(city)


class ArbitraryInsertion(_InsertionBase):
    name = "arbitrary_insertion"

    def solve(self, problem, *, start=0, initial_tour=None, **_):
        t0 = time.perf_counter()
        n = problem.n

        if initial_tour is not None:
            problem.validate_tour(initial_tour)
            tour = list(initial_tour)
            remaining = [c for c in range(n) if c not in set(tour)]
        else:
            tour = [0, 1]
            remaining = list(range(2, n))

        for c in remaining:
            _, pos, _ = self._best_insertion(problem, tour, [c])
            tour.insert(pos, c)

        return self._make_solution(problem, tour,
                                   time.perf_counter() - t0,
                                   start=start, mode="arbitrary")
