from tsp_studio import TSPProblem, get_algorithm
from tsp_studio.algorithms import _moves

DIST6 = [  # any symmetric 6x6 matrix
    [0, 12, 10, 19, 8, 14],
    [12, 0, 3, 7, 2, 9],
    [10, 3, 0, 6, 20, 5],
    [19, 7, 6, 0, 4, 11],
    [8, 2, 20, 4, 0, 15],
    [14, 9, 5, 11, 15, 0],
]


def run(tenure=2, iters=4):
    p = TSPProblem.from_matrix(DIST6)
    return p, get_algorithm("tabu_search")().solve(
        p, initial_tour=[0, 1, 2, 3, 4, 5], deterministic=True,
        record_log=True, max_iterations=iters, tabu_tenure=tenure)


def test_full_neighbourhood_is_enumerated_every_iteration():
    p, sol = run()
    log = sol.metadata["log"]
    assert len(log) == 4
    n_moves = len(_moves.all_moves(6))
    assert n_moves == 12              # segment length 2..4: 5 + 4 + 3 positions
    assert all(len(e["candidates"]) == n_moves for e in log)


def test_chosen_move_becomes_tabu_for_exactly_tenure_iterations():
    p, sol = run(tenure=2)
    log = sol.metadata["log"]
    m1 = log[0]["chosen"]
    assert log[0]["tabu_list"] == []
    assert (m1, 2) in log[1]["tabu_list"]      # 2 iterations left at iteration 2
    assert (m1, 1) in log[2]["tabu_list"]      # 1 left at iteration 3
    assert m1 not in [m for m, _ in log[3]["tabu_list"]]   # expired at iteration 4


def test_log_tours_and_lengths_are_consistent():
    p, sol = run()
    for e in sol.metadata["log"]:
        assert p.tour_length(e["tour_after"]) == pytest_approx(e["length"])
    log = sol.metadata["log"]
    assert log[0]["tour_after"] == log[1]["tour_before"]


def pytest_approx(x):
    import pytest
    return pytest.approx(x)
