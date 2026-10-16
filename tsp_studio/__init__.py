"""TSP Optimization Studio - Basic Algorithms Module."""
from .core.city import City
from .core.problem import TSPProblem
from .core.distance import METRICS, build_distance_matrix
from .algorithms.base import Solver, Solution
from .algorithms.pipeline import Pipeline
from .registry import (
    get_algorithm, list_algorithms, register,
    make_pipeline, all_combinations,
    CONSTRUCTION, IMPROVEMENT, METAHEURISTIC, EXACT, TEAMMATE,
)

__all__ = [
    "City", "TSPProblem", "METRICS", "build_distance_matrix",
    "Solver", "Solution", "Pipeline",
    "get_algorithm", "list_algorithms", "register",
    "make_pipeline", "all_combinations",
    "CONSTRUCTION", "IMPROVEMENT", "METAHEURISTIC", "EXACT", "TEAMMATE",
]
__version__ = "0.3.0"
