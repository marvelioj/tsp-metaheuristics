from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class City:
    id: int
    x: Optional[float] = None
    y: Optional[float] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.id < 0:
            raise ValueError(f"City id must be non-negative, got {self.id}")
        if (self.x is None) != (self.y is None):
            raise ValueError("City must have both x and y, or neither.")

    @property
    def has_coords(self) -> bool:
        return self.x is not None and self.y is not None

    def __repr__(self) -> str:
        label = self.name or f"City{self.id}"
        if self.has_coords:
            return f"{label}({self.x:.2f},{self.y:.2f})"
        return label
