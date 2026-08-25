from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from .models import FleetState
from .simulator import rollout, travel_hours


@dataclass(frozen=True)
class PCCMetrics:
    pressure: float
    entropy: float
    control: float
    instability: float
    cascade_risk: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pressure_score(state: FleetState) -> float:
    if not state.loads:
        return 0.0

    values: list[float] = []
    for load in state.loads:
        truck = state.truck(load.truck_id)
        if not truck.active:
            values.append(1.0)
            continue

        drive = (
            travel_hours((truck.x, truck.y), (load.pickup_x, load.pickup_y))
            + travel_hours((load.pickup_x, load.pickup_y), (load.dropoff_x, load.dropoff_y))
            + load.service_hours
        )
        schedule_pressure = clamp01(drive / max(load.due_in_hours, 0.1))
        hos_pressure = clamp01(drive / max(truck.available_hours, 0.1))
        values.append(0.58 * schedule_pressure + 0.42 * hos_pressure)

    active_ratio = sum(t.active for t in state.trucks) / max(len(state.trucks), 1)
    availability_pressure = 1.0 - active_ratio
    return clamp01(0.86 * (sum(values) / len(values)) + 0.14 * availability_pressure)


def shannon_entropy(labels: list[tuple[int, int]]) -> float:
    """Normalized entropy over coarse future outcome classes."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    probs = [c / n for c in counts.values()]
    h = -sum(p * math.log(p, 2) for p in probs)
    max_h = math.log(max(len(counts), 2), 2)
    return clamp01(h / max_h)


def evaluate_pcc(state: FleetState, rollouts: int = 250, seed: int = 7) -> PCCMetrics:
    rng = random.Random(seed)
    outcomes = [rollout(state, rng) for _ in range(rollouts)]

    # Coarsen outcomes so entropy describes how dispersed future operational states are.
    labels = [(o.late_loads, o.infeasible_loads) for o in outcomes]
    entropy = shannon_entropy(labels)
    pressure = pressure_score(state)

    max_options = max(len(state.loads) * max(len(state.trucks) - 1, 1), 1)
    avg_options = sum(o.remaining_control_options for o in outcomes) / max(len(outcomes), 1)
    control = clamp01(avg_options / max_options)

    cascade_failures = sum(1 for o in outcomes if (o.late_loads + o.infeasible_loads) >= 2)
    cascade_risk = cascade_failures / max(len(outcomes), 1)

    # Prototype instability index: high pressure + high entropy + low control.
    instability = clamp01(0.40 * pressure + 0.33 * entropy + 0.27 * (1.0 - control))

    return PCCMetrics(
        pressure=pressure,
        entropy=entropy,
        control=control,
        instability=instability,
        cascade_risk=cascade_risk,
    )
