import pytest
import pandas as pd

from tsp_studio.io.parsers import load_excel, load_coordinates_csv


def test_excel_coordinates(tmp_path):
    f = tmp_path / "c.xlsx"
    pd.DataFrame({"id": ["A", "B", "C", "D"], "X": [0, 1, 1, 0],
                  "Y": [0, 0, 1, 1]}).to_excel(f, index=False)
    p = load_excel(f)
    assert p.n == 4 and p.cities[0].name == "A"
    assert p.tour_length([0, 1, 2, 3]) == pytest.approx(4.0)


def test_excel_distance_matrix_plain_and_labelled(tmp_path):
    m = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    f1 = tmp_path / "m1.xlsx"
    pd.DataFrame(m).to_excel(f1, index=False, header=False)
    assert load_excel(f1).n == 3

    f2 = tmp_path / "m2.xlsx"
    df = pd.DataFrame(m, columns=["a", "b", "c"], index=["a", "b", "c"])
    df.to_excel(f2)                       # header row + label column
    p = load_excel(f2)
    assert p.n == 3 and p.dist(1, 2) == 3


def test_excel_errors_are_readable(tmp_path):
    bad = tmp_path / "bad.xlsx"
    pd.DataFrame({"x": [1, "oops", 3], "y": [1, 2, 3]}).to_excel(bad, index=False)
    with pytest.raises(ValueError, match="row"):
        load_excel(bad)

    few = tmp_path / "few.xlsx"
    pd.DataFrame({"x": [1, 2], "y": [1, 2]}).to_excel(few, index=False)
    with pytest.raises(ValueError, match="at least 3"):
        load_excel(few)

    dup = tmp_path / "dup.xlsx"
    pd.DataFrame({"id": ["a", "a", "b"], "x": [1, 2, 3], "y": [1, 2, 3]}).to_excel(dup, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_excel(dup)

    other = tmp_path / "other.xlsx"
    pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]}).to_excel(other, index=False)
    with pytest.raises(ValueError, match="Unrecognised"):
        load_excel(other)

    with pytest.raises(FileNotFoundError):
        load_excel(tmp_path / "nope.xlsx")


def test_csv_coordinates(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("id,x,y\n1,0,0\n2,3,0\n3,3,4\n")
    p = load_coordinates_csv(f)
    assert p.tour_length([0, 1, 2]) == pytest.approx(12.0)
