"""Shared 2-opt move helpers for the metaheuristics (SA, Tabu Search).

A move is a pair of positions (i, j) with i < j: the segment tour[i..j] is
reversed. Only moves whose segment length L = j - i + 1 satisfies
2 <= L <= n - 2 are valid; L = n - 1 and L = n only reverse the direction of
the whole cycle, so they change nothing.

The delta formulas assume a SYMMETRIC distance matrix.
"""
from __future__ import annotations
import random
from typing import Sequence


def require_symmetric(mat: Sequence[Sequence[float]], tol: float = 1e-9) -> None:
    n = len(mat)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(mat[i][j] - mat[j][i]) > tol:
                raise ValueError(
                    "This algorithm needs a symmetric distance matrix "
                    f"(d[{i}][{j}] != d[{j}][{i}]).")


def is_valid_move(i: int, j: int, n: int) -> bool:
    return 0 <= i < j < n and 2 <= j - i + 1 <= n - 2


def random_move(rng: random.Random, n: int) -> tuple[int, int]:
    """Random valid move (requires n >= 4)."""
    while True:
        i, j = sorted(rng.sample(range(n), 2))
        if is_valid_move(i, j, n):
            return i, j


def all_moves(n: int) -> list[tuple[int, int]]:
    """Every valid move, in lexicographic order (deterministic neighbourhood)."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if is_valid_move(i, j, n)]


def delta(tour: Sequence[int], i: int, j: int, mat) -> float:
    """Change in tour length if tour[i..j] is reversed (O(1))."""
    n = len(tour)
    a, b = tour[i - 1], tour[i]          # tour[-1] wraps around when i == 0
    c, d = tour[j], tour[(j + 1) % n]
    return mat[a][c] + mat[b][d] - mat[a][b] - mat[c][d]


def apply(tour: list[int], i: int, j: int) -> None:
    """Reverse tour[i..j] in place."""
    tour[i:j + 1] = tour[i:j + 1][::-1]
