from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
from .city import City
from .distance import build_distance_matrix, DistanceFn

MIN_CITIES = 3


@dataclass
class TSPProblem:
    cities: list[City]
    distance_matrix: list[list[float]]
    name: str = "unnamed"
    metric: str = "euclidean"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.cities)
        if n < MIN_CITIES:
            raise ValueError(f"TSP requires at least {MIN_CITIES} cities, got {n}.")
        if len(self.distance_matrix) != n:
            raise ValueError("Distance matrix size does not match cities.")
        for row in self.distance_matrix:
            if len(row) != n:
                raise ValueError("Distance matrix must be square.")

    @classmethod
    def from_cities(cls, cities: Sequence[City], *,
                    metric: str | DistanceFn = "euclidean",
                    name: str = "unnamed") -> "TSPProblem":
        cities = list(cities)
        mat = build_distance_matrix(cities, metric)
        return cls(cities=cities, distance_matrix=mat, name=name,
                   metric=metric if isinstance(metric, str) else "custom")

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]], *,
                    name: str = "unnamed") -> "TSPProblem":
        n = len(matrix)
        cities = [City(id=i, name=f"N{i}") for i in range(n)]
        mat = [list(row) for row in matrix]
        return cls(cities=cities, distance_matrix=mat, name=name, metric="matrix")

    @property
    def n(self) -> int:
        return len(self.cities)

    def dist(self, i: int, j: int) -> float:
        return self.distance_matrix[i][j]

    def tour_length(self, tour: Sequence[int]) -> float:
        if len(tour) < 2:
            return 0.0
        total = 0.0
        for k in range(len(tour)):
            total += self.distance_matrix[tour[k]][tour[(k + 1) % len(tour)]]
        return total

    def validate_tour(self, tour: Sequence[int]) -> None:
        if sorted(tour) != list(range(self.n)):
            raise ValueError("Tour must be a permutation of range(n).")

    def __repr__(self) -> str:
        return f"TSPProblem(name={self.name!r}, n={self.n}, metric={self.metric})"
