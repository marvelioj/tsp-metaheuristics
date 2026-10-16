from __future__ import annotations
import json
from pathlib import Path
from ..core.problem import TSPProblem


def save_problem_json(problem: TSPProblem, path: str | Path) -> None:
    payload = {
        "name": problem.name,
        "metric": problem.metric,
        "cities": [{"id": c.id, "x": c.x, "y": c.y, "name": c.name}
                   for c in problem.cities],
        "distance_matrix": problem.distance_matrix,
        "metadata": problem.metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_problem_json(path: str | Path) -> TSPProblem:
    data = json.loads(Path(path).read_text())
    from ..core.city import City
    cities = [City(id=c["id"], x=c.get("x"), y=c.get("y"), name=c.get("name"))
              for c in data["cities"]]
    return TSPProblem(cities=cities,
                      distance_matrix=data["distance_matrix"],
                      name=data.get("name", "unnamed"),
                      metric=data.get("metric", "matrix"),
                      metadata=data.get("metadata", {}))
