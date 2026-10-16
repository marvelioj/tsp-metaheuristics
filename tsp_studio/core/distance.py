from __future__ import annotations
import math
from typing import Callable, Sequence
from .city import City

DistanceFn = Callable[[City, City], float]


def euclidean(a: City, b: City) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def manhattan(a: City, b: City) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)


def pseudo_euclidean(a: City, b: City) -> float:
    r = math.sqrt(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) / 10.0)
    n = int(r + 0.5)
    return float(n + 1 if n < r else n)


def geographical(a: City, b: City) -> float:
    """TSPLIB GEO: coordinates are lat/long in DDD.MM format."""
    pi = 3.141592
    deg = int(a.x)
    min_ = a.x - deg
    lat_a = pi * (deg + 5.0 * min_ / 3.0) / 180.0
    deg = int(b.x)
    min_ = b.x - deg
    lat_b = pi * (deg + 5.0 * min_ / 3.0) / 180.0
    deg = int(a.y)
    min_ = a.y - deg
    lon_a = pi * (deg + 5.0 * min_ / 3.0) / 180.0
    deg = int(b.y)
    min_ = b.y - deg
    lon_b = pi * (deg + 5.0 * min_ / 3.0) / 180.0
    R = 6378.388
    q1 = math.cos(lon_a - lon_b)
    q2 = math.cos(lat_a - lat_b)
    q3 = math.cos(lat_a + lat_b)
    return int(R * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)


METRICS = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "pseudo_euclidean": pseudo_euclidean,
    "geographical": geographical,
}


def build_distance_matrix(cities: Sequence[City],
                          metric: str | DistanceFn = "euclidean") -> list[list[float]]:
    if callable(metric):
        fn = metric
    elif metric in METRICS:
        fn = METRICS[metric]
    else:
        raise ValueError(f"Unknown metric {metric!r}. Choices: {list(METRICS)}")

    n = len(cities)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not (cities[i].has_coords and cities[j].has_coords):
                raise ValueError("Distance metric requires coordinates for all cities.")
            d = fn(cities[i], cities[j])
            mat[i][j] = d
            mat[j][i] = d
    return mat
