import pytest

from tsp_studio import get_algorithm, TEAMMATE
from tsp_studio.io.generators import random_uniform


@pytest.fixture(scope="module")
def p20():
    return random_uniform(20, seed=42)


@pytest.fixture(scope="module")
def nn_tour(p20):
    return get_algorithm("nearest_neighbor")().solve(p20)


def valid(p, sol):
    return sorted(sol.tour) == list(range(p.n)) and sol.length == pytest.approx(p.tour_length(sol.tour))


def test_all_teammate_algorithms_are_registered():
    assert set(TEAMMATE) == {"farthest_insertion_far_start", "three_opt_swap",
                             "simulated_annealing_basic", "tabu_search_basic"}


def test_farthest_insertion_far_start_valid(p20):
    assert valid(p20, get_algorithm("farthest_insertion_far_start")().solve(p20))


def test_three_opt_swap_without_start_tour_does_not_crash(p20):
    # regression: the original imported a module that does not exist
    assert valid(p20, get_algorithm("three_opt_swap")().solve(p20))


def test_three_opt_swap_not_worse_than_start(p20, nn_tour):
    sol = get_algorithm("three_opt_swap")().solve(p20, initial_tour=nn_tour.tour)
    assert valid(p20, sol) and sol.length <= nn_tour.length + 1e-9


def test_sa_basic_valid_and_not_worse_than_start(p20, nn_tour):
    sol = get_algorithm("simulated_annealing_basic")().solve(
        p20, initial_tour=nn_tour.tour, t_max=10, t_min=0.1, alpha=0.9, iter_per_temp=20)
    assert valid(p20, sol) and sol.length <= nn_tour.length + 1e-9


def test_tabu_basic_valid_and_not_worse_than_start(p20, nn_tour):
    sol = get_algorithm("tabu_search_basic")().solve(
        p20, initial_tour=nn_tour.tour, max_iter=30)
    assert valid(p20, sol) and sol.length <= nn_tour.length + 1e-9
