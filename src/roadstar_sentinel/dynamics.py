from __future__ import annotations

from dataclasses import dataclass

from .models import FleetState, Load, Truck
from .pcc import PCCMetrics, evaluate_pcc
from .simulator import travel_hours


@dataclass(frozen=True)
class TrajectoryPoint:
    time_hours: float
    metrics: PCCMetrics
    entropy_rate: float
    pressure_rate: float
    early_warning: bool
    conventional_failure: bool


def deterministic_failure(state: FleetState) -> bool:
    """Conventional threshold: assigned work is already late or infeasible now."""
    failures = 0
    for load in state.loads:
        truck = state.truck(load.truck_id)
        if not truck.active:
            failures += 1
            continue
        required = (
            travel_hours((truck.x, truck.y), (load.pickup_x, load.pickup_y))
            + travel_hours((load.pickup_x, load.pickup_y), (load.dropoff_x, load.dropoff_y))
            + load.service_hours
        )
        if required > truck.available_hours or required > load.due_in_hours:
            failures += 1
    # A conventional alarm represents an already-obvious network-level service failure:
    # at least two assigned loads are simultaneously infeasible. A spare/failed truck by
    # itself is not a network failure if its work has already been reassigned.
    return failures >= 2


def degraded_state(base: FleetState, time_hours: float, failure_time: float = 2.0) -> FleetState:
    """Advance a synthetic fleet through a creeping T03 disruption.

    All deadlines and HOS budgets decay with time. T03 additionally loses control
    authority as a mechanical issue worsens, then fails at ``failure_time``.
    This gives the demo a pre-failure regime where uncertainty can rise before a
    conventional KPI reports a multi-load failure.
    """
    trucks: list[Truck] = []
    for truck in base.trucks:
        available = max(0.0, truck.available_hours - time_hours)
        active = truck.active
        if truck.id == "T03":
            # Progressive loss of usable HOS/capacity before the hard breakdown.
            degradation = 0.75 * time_hours + 0.35 * time_hours * time_hours
            available = max(0.0, available - degradation)
            if time_hours >= failure_time:
                active = False
                available = 0.0
        trucks.append(
            Truck(
                id=truck.id,
                x=truck.x,
                y=truck.y,
                available_hours=available,
                capacity=truck.capacity,
                active=active,
            )
        )

    loads = tuple(
        Load(
            id=load.id,
            truck_id=load.truck_id,
            pickup_x=load.pickup_x,
            pickup_y=load.pickup_y,
            dropoff_x=load.dropoff_x,
            dropoff_y=load.dropoff_y,
            due_in_hours=max(0.1, load.due_in_hours - time_hours),
            service_hours=load.service_hours,
            priority=load.priority,
        )
        for load in base.loads
    )
    return FleetState(time_hours=time_hours, trucks=tuple(trucks), loads=loads)



def sentinel_alarm(metrics: PCCMetrics, entropy_rate: float) -> bool:
    """Prototype PCC+entropy early-warning rule.

    Thresholds are intentionally explicit so the experiment harness can compare
    this detector against simpler ablations. They are synthetic demo thresholds,
    not validated fleet-safety limits.
    """
    return (
        (metrics.instability >= 0.56 and metrics.cascade_risk >= 0.25)
        or (entropy_rate >= 0.18 and metrics.pressure >= 0.50)
        or (metrics.entropy >= 0.62 and metrics.control <= 0.55)
    )

def simulate_trajectory(
    base: FleetState,
    duration_hours: float = 3.0,
    step_hours: float = 0.25,
    rollouts: int = 180,
    seed: int = 23,
) -> list[TrajectoryPoint]:
    raw: list[tuple[float, FleetState, PCCMetrics]] = []
    steps = int(round(duration_hours / step_hours))
    for i in range(steps + 1):
        t = round(i * step_hours, 10)
        state = degraded_state(base, t)
        metrics = evaluate_pcc(state, rollouts=rollouts, seed=seed + i)
        raw.append((t, state, metrics))

    points: list[TrajectoryPoint] = []
    for i, (t, state, metrics) in enumerate(raw):
        if i == 0:
            entropy_rate = 0.0
            pressure_rate = 0.0
        else:
            prev_t, _, prev = raw[i - 1]
            dt = max(t - prev_t, 1e-9)
            entropy_rate = (metrics.entropy - prev.entropy) / dt
            pressure_rate = (metrics.pressure - prev.pressure) / dt

        # Sentinel combines state level and rate-of-change information.
        # Thresholds are prototype/demo thresholds, not validated industry limits.
        early_warning = sentinel_alarm(metrics, entropy_rate)
        points.append(
            TrajectoryPoint(
                time_hours=t,
                metrics=metrics,
                entropy_rate=entropy_rate,
                pressure_rate=pressure_rate,
                early_warning=early_warning,
                conventional_failure=deterministic_failure(state),
            )
        )
    return points


def warning_lead_time(points: list[TrajectoryPoint]) -> float | None:
    warning = next((p.time_hours for p in points if p.early_warning), None)
    failure = next((p.time_hours for p in points if p.conventional_failure), None)
    if warning is None or failure is None or warning >= failure:
        return None
    return failure - warning
