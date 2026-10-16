"""Generate synthetic TSP datasets.

Run: python -m tsp_studio.scripts.dataset
"""
from __future__ import annotations
import json
from pathlib import Path
from ..io.generators import random_uniform, random_clustered, random_grid
from ..io.exporters import save_problem_json

OUT = Path("datasets")


def main():
    OUT.mkdir(exist_ok=True)
    specs = [
        ("uniform_n10",  random_uniform,   dict(n=10,  seed=1)),
        ("uniform_n20",  random_uniform,   dict(n=20,  seed=2)),
        ("uniform_n50",  random_uniform,   dict(n=50,  seed=3)),
        ("uniform_n100", random_uniform,   dict(n=100, seed=4)),
        ("uniform_n200", random_uniform,   dict(n=200, seed=5)),
        ("clustered_n30",  random_clustered, dict(n=30,  n_clusters=5,  seed=11)),
        ("clustered_n80",  random_clustered, dict(n=80,  n_clusters=8,  seed=12)),
        ("clustered_n150", random_clustered, dict(n=150, n_clusters=10, seed=13)),
        ("grid_n25",  random_grid, dict(n=25,  seed=21)),
        ("grid_n64",  random_grid, dict(n=64,  seed=22)),
        ("grid_n100", random_grid, dict(n=100, seed=23)),
    ]
    manifest = []
    for label, gen, kwargs in specs:
        problem = gen(**kwargs)
        path = OUT / f"{label}.json"
        save_problem_json(problem, path)
        manifest.append({
            "name": label, "n": problem.n,
            "metric": problem.metric, "file": path.as_posix(),
        })
        print(f"  OK  {label:20s} n={problem.n:4d}  ->  {path}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nGenerated {len(manifest)} datasets in {OUT.resolve()}")


if __name__ == "__main__":
    main()
