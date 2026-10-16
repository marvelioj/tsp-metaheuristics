"""Matplotlib figures for the UI (return a Figure; Streamlit: st.pyplot(fig))."""
from __future__ import annotations
from typing import Sequence

import matplotlib.pyplot as plt

from ..core.problem import TSPProblem


def plot_route_map(problem: TSPProblem, tour: Sequence[int],
                   title: str = "Rute") -> "plt.Figure":
    """Route on the plane. Needs coordinates (not available for matrix-only data)."""
    if not all(c.has_coords for c in problem.cities):
        raise ValueError("This instance has no coordinates (distance matrix only).")
    order = list(tour) + [tour[0]]
    xs = [problem.cities[i].x for i in order]
    ys = [problem.cities[i].y for i in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="#1f77b4", linewidth=2, markersize=7, label="Rute")
    start = problem.cities[tour[0]]
    ax.plot(start.x, start.y, "go", markersize=12, label="Start")
    for c in problem.cities:
        ax.text(c.x, c.y, f" {c.name or c.id}", fontsize=8, fontweight="bold")
    ax.set_title(f"{title}  (panjang = {problem.tour_length(tour):.2f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_convergence_curve(history_best: Sequence[float],
                           history_current: Sequence[float],
                           title: str = "Grafik Konvergensi") -> "plt.Figure":
    """Best vs current length over iterations (SA: per temperature level)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_current, color="#ff7f0e", alpha=0.4, label="Current (eksplorasi)")
    ax.plot(history_best, color="#2ca02c", linewidth=2.5, label="Best (eksploitasi)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Total jarak")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig
