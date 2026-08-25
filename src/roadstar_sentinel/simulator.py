from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .models import FleetState


@dataclass(frozen=True)
class RolloutOutcome:
    late_loads: int
    infeasible_loads: int
    weighted_lateness: float
    remaining_control_options: int


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def travel_hours(a: tuple[float, float], b: tuple[float, float], speed: float = 55.0) -> float:
    return euclidean(a, b) / speed


def rollout(state: FleetState, rng: random.Random, horizon_hours: float = 6.0) -> RolloutOutcome:
    """Lightweight stochastic future-state approximation.

    Coordinates are interpreted as miles in a synthetic plane. Each assigned load
    receives uncertain travel-time inflation to mimic congestion and disruption.
    """
    late = 0
    infeasible = 0
    weighted_lateness = 0.0
    remaining_options = 0

    active_trucks = [t for t in state.trucks if t.active]

    for load in state.loads:
        truck = state.truck(load.truck_id)
        if not truck.active:
            infeasible += 1
            weighted_lateness += 6.0 * load.priority
            continue

        to_pickup = travel_hours((truck.x, truck.y), (load.pickup_x, load.pickup_y))
        to_dropoff = travel_hours((load.pickup_x, load.pickup_y), (load.dropoff_x, load.dropoff_y))

        congestion = max(0.75, rng.gauss(1.0, 0.18))
        incident = 1.0 + (rng.uniform(0.15, 0.65) if rng.random() < 0.08 else 0.0)
        total_drive = (to_pickup + to_dropoff) * congestion * incident + load.service_hours

        if total_drive > truck.available_hours:
            infeasible += 1
            weighted_lateness += 4.0 * load.priority
        else:
            lateness = max(0.0, total_drive - load.due_in_hours)
            if lateness > 0:
                late += 1
                weighted_lateness += lateness * load.priority

        # Approximate control reserve: count feasible alternate active trucks.
        for alt in active_trucks:
            if alt.id == truck.id:
                continue
            alt_to_pickup = travel_hours((alt.x, alt.y), (load.pickup_x, load.pickup_y))
            if alt.available_hours >= alt_to_pickup + to_dropoff + load.service_hours:
                remaining_options += 1

    return RolloutOutcome(
        late_loads=late,
        infeasible_loads=infeasible,
        weighted_lateness=weighted_lateness,
        remaining_control_options=remaining_options,
    )
