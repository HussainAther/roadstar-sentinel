from __future__ import annotations

from math import hypot

from .interactive import IncidentKind, analyze_incident, incident_state
from .scenarios import creeping_failure_scenario

HUBS = (
    {"id": "IND", "label": "Indianapolis", "x": 22.0, "y": 18.0},
    {"id": "CHI", "label": "Chicago", "x": 50.0, "y": 72.0},
    {"id": "STL", "label": "St. Louis", "x": 7.0, "y": 52.0},
    {"id": "CMH", "label": "Columbus", "x": 89.0, "y": 47.0},
    {"id": "LOU", "label": "Louisville", "x": 45.0, "y": 5.0},
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _truck_position(truck, load, elapsed_hours: float) -> tuple[float, float, float]:
    """Simple deterministic route interpolation for the operations-center view."""
    if load is None:
        return truck.x, truck.y, 0.0
    total = max(load.due_in_hours + elapsed_hours, 0.5)
    progress = _clamp(elapsed_hours / total, 0.0, 0.88)
    # Move from truck origin toward the load dropoff; this is visualization state,
    # not a road-network routing claim.
    x = truck.x + (load.dropoff_x - truck.x) * progress
    y = truck.y + (load.dropoff_y - truck.y) * progress
    return x, y, progress


def _risk_for_assignment(state, truck, load) -> float:
    if not truck.active:
        return 1.0
    if load is None:
        return 0.12
    distance = hypot(truck.x - load.pickup_x, truck.y - load.pickup_y) + hypot(
        load.pickup_x - load.dropoff_x, load.pickup_y - load.dropoff_y
    )
    drive_hours = distance / 55.0 + load.service_hours
    schedule_pressure = _clamp((drive_hours - load.due_in_hours + 1.2) / 2.4, 0.0, 1.0)
    hos_pressure = _clamp((drive_hours - truck.available_hours + 1.0) / 2.0, 0.0, 1.0)
    return _clamp(0.58 * schedule_pressure + 0.42 * hos_pressure, 0.0, 1.0)


def _event_timeline(kind: IncidentKind, severity: float, analysis: dict) -> list[dict]:
    warning = analysis.get("warning_time_hours")
    failure = analysis.get("failure_time_hours")
    action = analysis.get("selected_action", {})
    events = [
        {
            "time_hours": 0.0,
            "type": "nominal",
            "title": "Fleet operating normally",
            "detail": "Sentinel establishes the baseline PCC state and future-state distribution.",
        },
        {
            "time_hours": 0.25,
            "type": "incident",
            "title": kind.replace("_", " ").title() + " begins",
            "detail": f"Synthetic disturbance enters the network at severity {severity:.1f}.",
        },
    ]
    if warning is not None:
        events.append(
            {
                "time_hours": warning,
                "type": "warning",
                "title": "Sentinel early warning",
                "detail": "Pressure/entropy dynamics cross the prototype early-warning rule before the hard failure threshold.",
            }
        )
        events.append(
            {
                "time_hours": warning,
                "type": "control",
                "title": "Control recommendation",
                "detail": action.get("label", "No intervention"),
            }
        )
    if failure is not None:
        events.append(
            {
                "time_hours": failure,
                "type": "failure",
                "title": "Conventional failure threshold",
                "detail": "A hard operational constraint is now visibly violated in the no-control trajectory.",
            }
        )
    return sorted(events, key=lambda x: (x["time_hours"], x["type"]))


def operations_view(kind: IncidentKind, severity: float = 0.7, time_hours: float | None = None) -> dict:
    severity = _clamp(severity, 0.0, 1.0)
    analysis = analyze_incident(kind, severity=severity, rollouts=55, seed=41)
    if time_hours is None:
        time_hours = analysis.get("warning_time_hours")
        if time_hours is None:
            time_hours = 1.0
    time_hours = _clamp(float(time_hours), 0.0, 3.0)

    base = creeping_failure_scenario()
    state = incident_state(base, kind, time_hours, severity)
    load_map = {load.truck_id: load for load in state.loads}

    trucks = []
    for truck in state.trucks:
        load = load_map.get(truck.id)
        x, y, progress = _truck_position(truck, load, time_hours)
        risk = _risk_for_assignment(state, truck, load)
        trucks.append(
            {
                "id": truck.id,
                "x": x,
                "y": y,
                "active": truck.active,
                "available_hours": truck.available_hours,
                "load_id": load.id if load else None,
                "risk": risk,
                "progress": progress,
                "status": "OUT" if not truck.active else "AT RISK" if risk >= 0.65 else "WATCH" if risk >= 0.38 else "OK",
            }
        )

    loads = [
        {
            "id": load.id,
            "truck_id": load.truck_id,
            "pickup": [load.pickup_x, load.pickup_y],
            "dropoff": [load.dropoff_x, load.dropoff_y],
            "due_in_hours": load.due_in_hours,
            "priority": load.priority,
        }
        for load in state.loads
    ]

    # Visual causal edges highlight which assignments are exposed by each incident.
    affected = {
        "truck_breakdown": ["L103"],
        "traffic_surge": ["L102", "L103", "L105"],
        "hos_shortage": ["L102", "L106"],
        "depot_outage": ["L104", "L106"],
        "compound_shock": ["L102", "L103", "L105", "L106"],
    }[kind]
    propagation = [
        {"from": kind, "to": lid, "strength": severity * (0.65 + 0.35 * min(1.0, time_hours / 1.5))}
        for lid in affected
    ]

    return {
        "incident": kind,
        "severity": severity,
        "time_hours": time_hours,
        "hubs": list(HUBS),
        "trucks": trucks,
        "loads": loads,
        "propagation": propagation,
        "timeline": _event_timeline(kind, severity, analysis),
        "warning_time_hours": analysis.get("warning_time_hours"),
        "failure_time_hours": analysis.get("failure_time_hours"),
        "selected_action": analysis.get("selected_action"),
        "disclaimer": "Synthetic schematic fleet map; coordinates are illustrative and not GPS or road-network data.",
    }
