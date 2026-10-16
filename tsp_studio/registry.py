"""Registry + pipeline helpers."""
from __future__ import annotations
from typing import Type
from .algorithms.base import Solver
from .algorithms.nearest_neighbor import NearestNeighbor
from .algorithms.insertion import (
    NearestInsertion, CheapestInsertion, FarthestInsertion, ArbitraryInsertion,
)
from .algorithms.two_opt import TwoOpt
from .algorithms.three_opt import ThreeOpt
from .algorithms.pipeline import Pipeline
from .algorithms.simulated_annealing import SimulatedAnnealing
from .algorithms.tabu_search import TabuSearch
from .algorithms.brute_force import HeldKarp
from .algorithms.teammate import (
    FarthestInsertionFarStart, ThreeOptSwap,
    SimulatedAnnealingBasic, TabuSearchBasic,
)


CONSTRUCTION = {
    "nearest_neighbor":    NearestNeighbor,
    "nearest_insertion":   NearestInsertion,
    "cheapest_insertion":  CheapestInsertion,
    "farthest_insertion":  FarthestInsertion,
    "arbitrary_insertion": ArbitraryInsertion,
}

IMPROVEMENT = {
    "two_opt":   TwoOpt,
    "three_opt": ThreeOpt,
}

METAHEURISTIC = {
    "simulated_annealing": SimulatedAnnealing,
    "tabu_search":         TabuSearch,
}

EXACT = {
    "held_karp_exact": HeldKarp,   # only for small n (<= 18)
}

# Kontribusi anggota tim (versi dasar) -- lihat algorithms/teammate.py
TEAMMATE = {
    "farthest_insertion_far_start": FarthestInsertionFarStart,
    "three_opt_swap":               ThreeOptSwap,
    "simulated_annealing_basic":    SimulatedAnnealingBasic,
    "tabu_search_basic":            TabuSearchBasic,
}

_ALGORITHMS: dict[str, Type[Solver]] = {
    **CONSTRUCTION, **IMPROVEMENT, **METAHEURISTIC, **EXACT, **TEAMMATE,
}


def register(cls):
    _ALGORITHMS[cls.name] = cls
    return cls


def get_algorithm(name: str):
    if name not in _ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}. "
                       f"Available: {sorted(_ALGORITHMS)}")
    return _ALGORITHMS[name]


def list_algorithms():
    return sorted(_ALGORITHMS)


def make_pipeline(*names: str, custom_name: str | None = None) -> Pipeline:
    """Build a Pipeline from a list of algorithm names."""
    solvers = [get_algorithm(n)() for n in names]
    return Pipeline(solvers, name=custom_name)


def all_combinations() -> dict[str, Pipeline]:
    """All 20 combinations: 5 construction x (none, 2opt, 3opt, 2opt+3opt)."""
    combos = {}
    for cname in CONSTRUCTION:
        combos[cname] = make_pipeline(cname)
        combos[f"{cname}+2opt"] = make_pipeline(cname, "two_opt")
        combos[f"{cname}+3opt"] = make_pipeline(cname, "three_opt")
        combos[f"{cname}+2opt+3opt"] = make_pipeline(cname, "two_opt", "three_opt")
    return combos
