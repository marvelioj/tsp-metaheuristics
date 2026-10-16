from __future__ import annotations
import random
from typing import Optional
from ..core.city import City
from ..core.problem import TSPProblem


def random_uniform(n: int, *, seed: Optional[int] = None,
                   width: float = 100.0, height: float = 100.0) -> TSPProblem:
    if n < 3:
        raise ValueError("At least 3 cities required.")
    rng = random.Random(seed)
    cities = [City(id=i, x=rng.uniform(0, width), y=rng.uniform(0, height))
              for i in range(n)]
    return TSPProblem.from_cities(cities, name=f"uniform_{n}_seed{seed}")


def random_clustered(n: int, n_clusters: int = 5, *,
                     seed: Optional[int] = None,
                     spread: float = 5.0,
                     width: float = 100.0,
                     height: float = 100.0) -> TSPProblem:
    if n < 3:
        raise ValueError("At least 3 cities required.")
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    rng = random.Random(seed)
    centers = [(rng.uniform(0, width), rng.uniform(0, height))
               for _ in range(n_clusters)]
    cities = []
    for i in range(n):
        cx, cy = centers[rng.randrange(n_clusters)]
        cities.append(City(id=i,
                           x=min(max(rng.gauss(cx, spread), 0), width),
                           y=min(max(rng.gauss(cy, spread), 0), height)))
    return TSPProblem.from_cities(cities, name=f"clustered_{n}x{n_clusters}_seed{seed}")


def random_grid(n: int, *, seed: Optional[int] = None) -> TSPProblem:
    if n < 3:
        raise ValueError("At least 3 cities required.")
    rng = random.Random(seed)
    side = int(n ** 0.5) + 1
    cities = []
    for i in range(n):
        r, c = divmod(i, side)
        cities.append(City(id=i,
                           x=c * 10.0 + rng.uniform(-0.5, 0.5),
                           y=r * 10.0 + rng.uniform(-0.5, 0.5)))
    return TSPProblem.from_cities(cities, name=f"grid_{n}_seed{seed}")
