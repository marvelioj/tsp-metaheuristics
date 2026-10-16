import math
import pytest

from tsp_studio import (City, TSPProblem, get_algorithm, make_pipeline,
                        CONSTRUCTION, IMPROVEMENT, METAHEURISTIC)
from tsp_studio.io.generators import random_uniform


@pytest.fixture(scope="module")
def p20():
    return random_uniform(20, seed=42)


def test_tour_length_unit_square():
    cities = [City(0, 0, 0), City(1, 1, 0), City(2, 1, 1), City(3, 0, 1)]
    p = TSPProblem.from_cities(cities)
    assert p.tour_length([0, 1, 2, 3]) == pytest.approx(4.0)
    assert p.tour_length([0, 2, 1, 3]) == pytest.approx(2 + 2 * math.sqrt(2))


@pytest.mark.parametrize("name", sorted({**CONSTRUCTION, **IMPROVEMENT, **METAHEURISTIC}))
def test_every_algorithm_returns_valid_tour(name, p20):
    sol = get_algorithm(name)().solve(p20, seed=1)
    assert sorted(sol.tour) == list(range(p20.n))
    assert sol.length == pytest.approx(p20.tour_length(sol.tour))


def test_two_opt_not_worse_than_start(p20):
    nn = get_algorithm("nearest_neighbor")().solve(p20)
    improved = get_algorithm("two_opt")().solve(p20, initial_tour=nn.tour)
    assert improved.length <= nn.length + 1e-9


def test_held_karp_is_optimal_on_small_instance():
    p = random_uniform(8, seed=3)
    exact = get_algorithm("held_karp_exact")().solve(p)
    for name in ["nearest_neighbor", "two_opt", "simulated_annealing", "tabu_search"]:
        other = get_algorithm(name)().solve(p, seed=1)
        assert exact.length <= other.length + 1e-9


@pytest.mark.parametrize("name", ["simulated_annealing", "tabu_search"])
def test_metaheuristic_seed_reproducible_and_never_worse(name, p20):
    a = get_algorithm(name)().solve(p20, seed=5)
    b = get_algorithm(name)().solve(p20, seed=5)
    assert a.tour == b.tour and a.length == b.length
    ins = a.metadata["insights"]
    assert a.length <= ins["initial_distance"] + 1e-9
    # incrementally tracked best == length recomputed from the tour
    assert ins["best_distance"] == pytest.approx(a.length)
    assert a.metadata["initial_tour"] is not None
    assert len(a.metadata["history_best"]) > 1


def test_sa_parameter_validation(p20):
    sa = get_algorithm("simulated_annealing")()
    with pytest.raises(ValueError):
        sa.solve(p20, alpha=1.5)
    with pytest.raises(ValueError):
        sa.solve(p20, T_min=10, T_max=1)


def test_small_instances_do_not_crash():
    p3 = random_uniform(3, seed=1)
    for name in ["simulated_annealing", "tabu_search"]:
        sol = get_algorithm(name)().solve(p3, seed=1)
        assert sorted(sol.tour) == [0, 1, 2]
    p4 = random_uniform(4, seed=1)
    for name in ["simulated_annealing", "tabu_search"]:
        assert sorted(get_algorithm(name)().solve(p4, seed=1).tour) == [0, 1, 2, 3]


def test_asymmetric_matrix_rejected():
    p = TSPProblem.from_matrix([[0, 1, 2, 3], [9, 0, 1, 2], [2, 9, 0, 1], [1, 2, 9, 0]])
    with pytest.raises(ValueError):
        get_algorithm("tabu_search")().solve(p)


def test_pipeline_construction_then_metaheuristic_keeps_history(p20):
    pipe = make_pipeline("nearest_neighbor", "tabu_search")
    sol = pipe.solve(p20, seed=1, max_iterations=50)
    assert [s["solver"] for s in sol.metadata["steps"]] == ["nearest_neighbor", "tabu_search"]
    assert "history_best" in sol.metadata and "insights" in sol.metadata
    # tabu started from the NN tour, not from a random one
    nn = get_algorithm("nearest_neighbor")().solve(p20)
    assert sol.metadata["initial_tour"] == nn.tour
    assert sol.length <= nn.length + 1e-9
