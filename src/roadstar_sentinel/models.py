from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class Truck:
    id: str
    x: float
    y: float
    available_hours: float
    capacity: float = 1.0
    active: bool = True


@dataclass(frozen=True)
class Load:
    id: str
    truck_id: str
    pickup_x: float
    pickup_y: float
    dropoff_x: float
    dropoff_y: float
    due_in_hours: float
    service_hours: float = 0.25
    priority: float = 1.0


@dataclass(frozen=True)
class FleetState:
    time_hours: float
    trucks: tuple[Truck, ...]
    loads: tuple[Load, ...]

    def truck(self, truck_id: str) -> Truck:
        return next(t for t in self.trucks if t.id == truck_id)

    def load(self, load_id: str) -> Load:
        return next(l for l in self.loads if l.id == load_id)

    def replace_truck(self, updated: Truck) -> "FleetState":
        trucks = tuple(updated if t.id == updated.id else t for t in self.trucks)
        return replace(self, trucks=trucks)

    def replace_load(self, updated: Load) -> "FleetState":
        loads = tuple(updated if l.id == updated.id else l for l in self.loads)
        return replace(self, loads=loads)


ControlKind = Literal["reassign", "reserve", "delay", "none"]


@dataclass(frozen=True)
class ControlAction:
    kind: ControlKind
    load_id: str | None = None
    target_truck_id: str | None = None
    delay_hours: float = 0.0
    label: str = "No action"
