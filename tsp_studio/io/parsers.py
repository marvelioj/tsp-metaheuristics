"""Parsers for common TSP input formats."""
from __future__ import annotations
import csv
import math
from pathlib import Path
from ..core.city import City
from ..core.problem import TSPProblem


def load_coordinates_csv(path: str | Path, *,
                         has_header: bool = True,
                         name: str | None = None) -> TSPProblem:
    """CSV with columns: id,x,y (extra columns ignored)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            rows.append((int(row[0]), float(row[1]), float(row[2])))
    cities = [City(id=i, x=x, y=y) for i, x, y in rows]
    return TSPProblem.from_cities(cities, name=name or Path(path).stem)


def load_distance_matrix_csv(path: str | Path, *,
                             name: str | None = None) -> TSPProblem:
    """CSV of a square distance matrix (no header)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        matrix = [[float(v) for v in row if v.strip()] for row in reader if row]
    return TSPProblem.from_matrix(matrix, name=name or Path(path).stem)


def load_tsplib(path: str | Path) -> TSPProblem:
    """Minimal TSPLIB parser supporting EUC_2D, ATT, GEO, and EXPLICIT."""
    text = Path(path).read_text()
    name = "tsplib"
    edge_weight_type = "EUC_2D"
    dimension = None
    coords: dict[int, tuple[float, float]] = {}
    matrix_rows: list[list[float]] = []
    section = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        upper = line.upper()
        if upper.startswith("NAME"):
            name = line.split(":", 1)[1].strip()
        elif upper.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1].strip())
        elif upper.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip().upper()
        elif upper.startswith("NODE_COORD_SECTION"):
            section = "coords"
        elif upper.startswith("EDGE_WEIGHT_SECTION"):
            section = "matrix"
        elif upper.startswith("EOF"):
            break
        elif section == "coords":
            parts = line.split()
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "matrix":
            matrix_rows.extend(float(v) for v in line.split())

    if section == "matrix" and dimension:
        mat = [matrix_rows[i * dimension:(i + 1) * dimension]
               for i in range(dimension)]
        return TSPProblem.from_matrix(mat, name=name)

    if not coords:
        raise ValueError("TSPLIB file contained no recognizable data.")

    cities = [City(id=i - 1, x=xy[0], y=xy[1])
              for i, xy in sorted(coords.items())]
    metric_map = {"EUC_2D": "euclidean", "ATT": "pseudo_euclidean", "GEO": "geographical"}
    metric = metric_map.get(edge_weight_type, "euclidean")
    return TSPProblem.from_cities(cities, metric=metric, name=name)



def load_excel(path: str | Path, *, sheet_name: int | str = 0,
               metric: str = "euclidean",
               name: str | None = None) -> TSPProblem:
    """Load a TSP instance from an Excel file (.xlsx).

    Two layouts are recognised (auto-detected):
      1. Coordinates: a header row with columns `x` and `y`; optional `id` or
         `name` column (labels only). Row order defines the city index 0..n-1.
      2. Distance matrix: an n x n block of numbers, optionally with a header
         row and a label column.

    Raises ValueError with a readable message when the sheet is invalid.
    """
    import pandas as pd   # imported here so the core package works without pandas

    path = Path(path)
    name = name or path.stem
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001 - unreadable / corrupt file
        raise ValueError(f"Cannot read Excel file {path.name}: {e}") from e

    df = df.dropna(how="all")
    cols = {str(c).strip().lower(): c for c in df.columns}

    if "x" in cols and "y" in cols:
        xs = pd.to_numeric(df[cols["x"]], errors="coerce")
        ys = pd.to_numeric(df[cols["y"]], errors="coerce")
        bad = df.index[xs.isna() | ys.isna()]
        if len(bad):
            rows = ", ".join(str(int(i) + 2) for i in bad[:5])   # +2: header + 1-based
            raise ValueError(f"Non-numeric or empty x/y in Excel row(s): {rows}")
        label_col = cols.get("name", cols.get("id"))
        if label_col is not None and df[label_col].duplicated().any():
            raise ValueError(f"Duplicate values in column '{label_col}'.")
        labels = (df[label_col].astype(str).tolist() if label_col is not None
                  else [None] * len(df))
        cities = [City(id=i, x=float(x), y=float(y), name=lab)
                  for i, (x, y, lab) in enumerate(zip(xs, ys, labels))]
        if len(cities) < 3:
            raise ValueError(f"Need at least 3 cities, found {len(cities)}.")
        return TSPProblem.from_cities(cities, metric=metric, name=name)

    # --- distance-matrix layout ---
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None).dropna(how="all")
    raw = raw.dropna(how="all", axis=1)
    num = raw.apply(pd.to_numeric, errors="coerce")
    for block in (num, num.iloc[1:, 1:]):          # plain, then with labels
        if block.shape[0] == block.shape[1] >= 3 and block.notna().all().all():
            matrix = block.to_numpy(dtype=float).tolist()
            if any(v < 0 or math.isnan(v) for row in matrix for v in row):
                raise ValueError("Distance matrix contains negative values.")
            return TSPProblem.from_matrix(matrix, name=name)
    raise ValueError(
        "Unrecognised layout. Use either columns 'x' and 'y' (optional 'id'/'name'), "
        "or a square n x n distance matrix.")
