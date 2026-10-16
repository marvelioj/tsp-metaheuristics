"""Run the algorithms on every dataset in datasets/manifest.json and print a table.

Run from the repo root:  python -m tsp_studio.scripts.benchmark
Cells show the tour length; '--' = skipped because the instance is too large
for that algorithm, 'ERR' = the algorithm raised an exception.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..io.exporters import load_problem_json
from ..registry import get_algorithm

ALGO_ORDER = [
    "nearest_neighbor",
    "cheapest_insertion",
    "nearest_insertion",
    "farthest_insertion",
    "two_opt",
    "three_opt",
    "simulated_annealing",
    "tabu_search",
    # versi dasar dari anggota tim (algorithms/teammate.py)
    "farthest_insertion_far_start",
    "three_opt_swap",
    "simulated_annealing_basic",
    "tabu_search_basic",
    "held_karp_exact",
]

# short column titles for the table
LABELS = {
    "nearest_neighbor": "NN", "cheapest_insertion": "cheapest",
    "nearest_insertion": "nearest", "farthest_insertion": "farthest",
    "two_opt": "2opt", "three_opt": "3opt",
    "simulated_annealing": "SA", "tabu_search": "Tabu",
    "farthest_insertion_far_start": "far_start", "three_opt_swap": "3opt_swap",
    "simulated_annealing_basic": "SA_basic", "tabu_search_basic": "Tabu_basic",
    "held_karp_exact": "HeldKarp",
}

# size limits (3-opt is O(n^3) with full length recomputation; Held-Karp is exponential)
MAX_N = {"held_karp_exact": 12, "three_opt": 50,
         "three_opt_swap": 50, "simulated_annealing_basic": 50}
SEED = 1   # SA / Tabu are random: fixed seed -> reproducible table


def run_one(problem, algo_name):
    try:
        solver = get_algorithm(algo_name)()
        return solver.solve(problem, seed=SEED)
    except Exception as e:  # noqa: BLE001
        return e


def main():
    datasets_dir = Path("datasets")
    manifest_path = datasets_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Run: python -m tsp_studio.scripts.dataset")
        return

    manifest = json.loads(manifest_path.read_text())
    width = 11
    header = f"{'dataset':16s} {'n':>4s}  " + \
             "  ".join(f"{LABELS[a]:>{width}s}" for a in ALGO_ORDER)
    print(header)
    print("-" * len(header))

    for entry in manifest:
        # tolerate Windows-style separators in older manifests
        problem = load_problem_json(entry["file"].replace("\\", "/"))
        row = [f"{entry['name']:16s}", f"{problem.n:4d}"]
        for algo in ALGO_ORDER:
            if problem.n > MAX_N.get(algo, 10**9):
                row.append(f"{'--':>{width}s}")
                continue
            result = run_one(problem, algo)
            row.append(f"{'ERR':>{width}s}" if isinstance(result, Exception)
                       else f"{result.length:{width}.2f}")
        print("  ".join(row), flush=True)


if __name__ == "__main__":
    main()
