from __future__ import annotations

from .models import FleetState, Load, Truck


def baseline_scenario() -> FleetState:
    trucks = (
        Truck("T01", 0, 0, 6.0),
        Truck("T02", 35, 15, 5.5),
        Truck("T03", 70, -10, 4.8),
        Truck("T04", 20, 60, 6.5),
        Truck("T05", 95, 45, 5.0),
        Truck("T06", 55, 55, 3.8),
    )
    loads = (
        Load("L101", "T01", 5, 5, 120, 10, 2.7, priority=1.3),
        Load("L102", "T02", 30, 20, 145, 45, 3.0, priority=1.1),
        Load("L103", "T03", 70, -5, 160, 20, 2.5, priority=1.4),
        Load("L104", "T04", 25, 60, 130, 80, 2.6, priority=1.0),
        Load("L105", "T05", 95, 50, 185, 40, 2.4, priority=1.2),
        Load("L106", "T06", 55, 50, 150, 70, 2.0, priority=1.5),
    )
    return FleetState(time_hours=0.0, trucks=trucks, loads=loads)


def disrupted_scenario() -> FleetState:
    state = baseline_scenario()
    # T03 breakdown: a local disturbance with potential downstream consequences.
    broken = state.truck("T03")
    state = state.replace_truck(
        Truck(
            id=broken.id,
            x=broken.x,
            y=broken.y,
            available_hours=0.0,
            capacity=broken.capacity,
            active=False,
        )
    )
    return state


def creeping_failure_scenario() -> FleetState:
    """Relaxed starting point for the dynamic early-warning demonstration.

    The fleet begins with genuine schedule/HOS reserve, then a degrading T03
    condition consumes that reserve over time. This separates an early-warning
    regime from the later, obvious hard failure.
    """
    base = baseline_scenario()
    trucks = tuple(
        Truck(
            id=t.id,
            x=t.x,
            y=t.y,
            available_hours=t.available_hours + 1.5,
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
            due_in_hours=l.due_in_hours + 1.25,
            service_hours=l.service_hours,
            priority=l.priority,
        )
        for l in base.loads
    )
    return FleetState(time_hours=0.0, trucks=trucks, loads=loads)
