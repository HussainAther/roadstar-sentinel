from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .controls import apply_action, candidate_actions
from .dynamics import TrajectoryPoint, deterministic_failure, sentinel_alarm
from .models import ControlAction, FleetState, Load, Truck
from .pcc import evaluate_pcc
from .scenarios import creeping_failure_scenario

IncidentKind = Literal[
    "truck_breakdown",
    "traffic_surge",
    "hos_shortage",
    "depot_outage",
    "compound_shock",
]


@dataclass(frozen=True)
class IncidentDefinition:
    kind: IncidentKind
    label: str
    description: str
    operational_effect: str


INCIDENTS: tuple[IncidentDefinition, ...] = (
    IncidentDefinition(
        "truck_breakdown",
        "Truck breakdown",
        "T03 develops a progressive mechanical problem and eventually becomes unavailable.",
        "Consumes T03 control authority and makes L103 increasingly fragile.",
    ),
    IncidentDefinition(
        "traffic_surge",
        "Traffic surge",
        "A corridor slowdown erodes schedule margin on L102, L103, and L105.",
        "Raises schedule pressure and broadens future ETA outcomes without immediately removing a truck.",
    ),
    IncidentDefinition(
        "hos_shortage",
        "HOS shortage",
        "Available driver hours fall faster than planned on T02 and T06.",
        "Reduces feasible recovery options while assigned work continues consuming time.",
    ),
    IncidentDefinition(
        "depot_outage",
        "Depot outage",
        "A synthetic regional depot outage progressively removes T04/T06 support capacity.",
        "Proxy for lost staging/dispatch capacity in this simplified fleet model.",
    ),
    IncidentDefinition(
        "compound_shock",
        "Compound shock",
        "A T03 mechanical issue coincides with corridor congestion and HOS pressure.",
        "Creates interacting disturbances that can amplify one another across assignments.",
    ),
)


def incident_catalog() -> list[dict]:
    return [
        {
            "kind": i.kind,
            "label": i.label,
            "description": i.description,
            "operational_effect": i.operational_effect,
        }
        for i in INCIDENTS
    ]


def _replace_trucks(state: FleetState, updates: dict[str, tuple[float, bool]]) -> FleetState:
    trucks: list[Truck] = []
    for truck in state.trucks:
        available, active = updates.get(truck.id, (truck.available_hours, truck.active))
        trucks.append(
            Truck(
                id=truck.id,
                x=truck.x,
                y=truck.y,
                available_hours=max(0.0, available),
                capacity=truck.capacity,
                active=active,
            )
        )
    return FleetState(time_hours=state.time_hours, trucks=tuple(trucks), loads=state.loads)


def _replace_loads(state: FleetState, due_adjustments: dict[str, float]) -> FleetState:
    loads: list[Load] = []
    for load in state.loads:
        adjustment = due_adjustments.get(load.id, 0.0)
        loads.append(
            Load(
                id=load.id,
                truck_id=load.truck_id,
                pickup_x=load.pickup_x,
                pickup_y=load.pickup_y,
                dropoff_x=load.dropoff_x,
                dropoff_y=load.dropoff_y,
                due_in_hours=max(0.1, load.due_in_hours - adjustment),
                service_hours=load.service_hours,
                priority=load.priority,
            )
        )
    return FleetState(time_hours=state.time_hours, trucks=state.trucks, loads=tuple(loads))


def incident_state(
    base: FleetState,
    kind: IncidentKind,
    elapsed_hours: float,
    severity: float = 0.7,
) -> FleetState:
    """Advance the fleet and inject one synthetic disturbance family.

    Severity is normalized to [0, 1]. The disturbance ramps with elapsed time so
    the dashboard can show a transition from nominal operation into a stressed
    regime rather than an instantaneous hard failure.
    """
    severity = max(0.0, min(1.0, severity))

    # Normal passage of time consumes both deadlines and HOS budgets.
    trucks = tuple(
        Truck(
            id=t.id,
            x=t.x,
            y=t.y,
            available_hours=max(0.0, t.available_hours - elapsed_hours),
            capacity=t.capacity,
            active=t.active,
        )
        for t in base.trucks
    )
    loads = tuple(
        Load(
            id=l.id,
            truck_id=l.truck_id,
            pickup_x=l.pickup_x,
            pickup_y=l.pickup_y,
            dropoff_x=l.dropoff_x,
            dropoff_y=l.dropoff_y,
            due_in_hours=max(0.1, l.due_in_hours - elapsed_hours),
            service_hours=l.service_hours,
            priority=l.priority,
        )
        for l in base.loads
    )
    state = FleetState(time_hours=elapsed_hours, trucks=trucks, loads=loads)

    ramp = min(1.0, elapsed_hours / 1.75)
    strength = severity * ramp

    if kind in {"truck_breakdown", "compound_shock"}:
        t = state.truck("T03")
        loss = strength * (1.2 + 1.5 * elapsed_hours)
        active = t.active and not (severity >= 0.55 and elapsed_hours >= (2.35 - 0.9 * severity))
        state = _replace_trucks(state, {"T03": (t.available_hours - loss, active)})

    if kind in {"traffic_surge", "compound_shock"}:
        # Congestion is represented as disappearing schedule margin on the corridor.
        traffic_loss = strength * (0.45 + 0.55 * elapsed_hours)
        state = _replace_loads(
            state,
            {"L102": 0.85 * traffic_loss, "L103": traffic_loss, "L105": 0.70 * traffic_loss},
        )

    if kind in {"hos_shortage", "compound_shock"}:
        loss = strength * (0.65 + 0.8 * elapsed_hours)
        t2 = state.truck("T02")
        t6 = state.truck("T06")
        state = _replace_trucks(
            state,
            {
                "T02": (t2.available_hours - loss, t2.active),
                "T06": (t6.available_hours - 1.15 * loss, t6.active),
            },
        )

    if kind == "depot_outage":
        # The compact demo has no explicit depot object. T04/T06 represent support
        # capacity from one synthetic staging region; the outage progressively
        # removes that regional control authority.
        t4 = state.truck("T04")
        t6 = state.truck("T06")
        loss = strength * (0.8 + elapsed_hours)
        cutoff = 2.1 - 0.75 * severity
        state = _replace_trucks(
            state,
            {
                "T04": (t4.available_hours - loss, t4.active and elapsed_hours < cutoff),
                "T06": (t6.available_hours - 0.7 * loss, t6.active),
            },
        )

    return state


def simulate_incident(
    kind: IncidentKind,
    severity: float = 0.7,
    duration_hours: float = 3.0,
    step_hours: float = 0.25,
    rollouts: int = 90,
    seed: int = 41,
    base: FleetState | None = None,
) -> list[TrajectoryPoint]:
    base = base or creeping_failure_scenario()
    raw: list[tuple[float, FleetState, object]] = []
    steps = int(round(duration_hours / step_hours))
    for i in range(steps + 1):
        t = round(i * step_hours, 10)
        state = incident_state(base, kind, t, severity)
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
        points.append(
            TrajectoryPoint(
                time_hours=t,
                metrics=metrics,
                entropy_rate=entropy_rate,
                pressure_rate=pressure_rate,
                early_warning=sentinel_alarm(metrics, entropy_rate),
                conventional_failure=deterministic_failure(state),
            )
        )
    return points


def _trajectory_with_action(
    base: FleetState,
    kind: IncidentKind,
    severity: float,
    action: ControlAction,
    control_time: float,
    reference: list[TrajectoryPoint],
    rollouts: int,
    seed: int,
) -> list[TrajectoryPoint]:
    controlled_base = apply_action(base, action)
    points: list[TrajectoryPoint] = []
    prev_entropy: float | None = None
    prev_pressure: float | None = None
    prev_time: float | None = None

    for i, original in enumerate(reference):
        t = original.time_hours
        if t < control_time:
            points.append(original)
            prev_entropy = original.metrics.entropy
            prev_pressure = original.metrics.pressure
            prev_time = t
            continue

        state = incident_state(controlled_base, kind, t, severity)
        metrics = evaluate_pcc(state, rollouts=rollouts, seed=seed + i)
        if prev_time is None:
            erate = prate = 0.0
        else:
            dt = max(t - prev_time, 1e-9)
            erate = (metrics.entropy - (prev_entropy or 0.0)) / dt
            prate = (metrics.pressure - (prev_pressure or 0.0)) / dt
        points.append(
            TrajectoryPoint(
                time_hours=t,
                metrics=metrics,
                entropy_rate=erate,
                pressure_rate=prate,
                early_warning=sentinel_alarm(metrics, erate),
                conventional_failure=deterministic_failure(state),
            )
        )
        prev_entropy = metrics.entropy
        prev_pressure = metrics.pressure
        prev_time = t
    return points


def _trajectory_cost(points: list[TrajectoryPoint], control_time: float, action: ControlAction) -> float:
    post = [p for p in points if p.time_hours >= control_time]
    instability = mean(p.metrics.instability for p in post)
    cascade = mean(p.metrics.cascade_risk for p in post)
    failure_fraction = sum(p.conventional_failure for p in post) / max(len(post), 1)
    action_penalty = 0.0 if action.kind == "none" else 0.015
    return instability + 0.35 * cascade + 0.25 * failure_fraction + action_penalty


def analyze_incident(
    kind: IncidentKind,
    severity: float = 0.7,
    rollouts: int = 90,
    seed: int = 41,
) -> dict:
    base = creeping_failure_scenario()
    no_control = simulate_incident(kind, severity=severity, rollouts=rollouts, seed=seed, base=base)
    warning = next((p for p in no_control if p.early_warning), None)
    failure = next((p for p in no_control if p.conventional_failure), None)

    if warning is None:
        # Low-severity runs can stay stable. Evaluate the final state so the UI still
        # offers useful contingency actions instead of throwing an error.
        control_time = no_control[-1].time_hours
        warning_state = incident_state(base, kind, control_time, severity)
    else:
        control_time = warning.time_hours
        warning_state = incident_state(base, kind, control_time, severity)

    actions = candidate_actions(warning_state, max_candidates=16)
    scored: list[tuple[float, ControlAction, list[TrajectoryPoint]]] = []
    for action in actions:
        trajectory = _trajectory_with_action(
            base,
            kind,
            severity,
            action,
            control_time,
            no_control,
            rollouts=max(35, rollouts // 2),
            seed=seed + 100,
        )
        scored.append((_trajectory_cost(trajectory, control_time, action), action, trajectory))
    scored.sort(key=lambda x: x[0])
    best_cost, best_action, controlled = scored[0]
    no_action = next((x for x in scored if x[1].kind == "none"), scored[0])

    selected_point = next((p for p in controlled if p.time_hours >= control_time), controlled[-1])
    lead = None
    if warning is not None and failure is not None and warning.time_hours < failure.time_hours:
        lead = failure.time_hours - warning.time_hours

    return {
        "incident": kind,
        "severity": severity,
        "warning_time_hours": warning.time_hours if warning else None,
        "failure_time_hours": failure.time_hours if failure else None,
        "warning_lead_time_hours": lead,
        "status": (
            "CRITICAL" if failure is not None else "EARLY WARNING" if warning is not None else "STABLE / WATCH"
        ),
        "selected_action": {
            **best_action.__dict__,
            "objective": best_cost,
            "improvement_vs_no_action": max(0.0, no_action[0] - best_cost),
        },
        "recommendations": [
            {**action.__dict__, "objective": cost}
            for cost, action, _ in scored[:5]
        ],
        "current_metrics": selected_point.metrics.__dict__,
        "no_control": no_control,
        "sentinel_control": controlled,
    }
