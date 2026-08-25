from __future__ import annotations

from dataclasses import dataclass

from .models import ControlAction, FleetState, Load, Truck
from .pcc import PCCMetrics, evaluate_pcc
from .simulator import travel_hours


@dataclass(frozen=True)
class RankedAction:
    action: ControlAction
    metrics: PCCMetrics
    improvement: float


def candidate_actions(state: FleetState, max_candidates: int = 24) -> list[ControlAction]:
    actions = [ControlAction(kind="none", label="No intervention")]

    for load in state.loads:
        current = state.truck(load.truck_id)
        actions.append(
            ControlAction(
                kind="reserve",
                load_id=load.id,
                label=f"Dispatch reserve unit for {load.id}",
            )
        )
        if len(actions) >= max_candidates:
            return actions
        for truck in state.trucks:
            if not truck.active or truck.id == current.id:
                continue
            needed = (
                travel_hours((truck.x, truck.y), (load.pickup_x, load.pickup_y))
                + travel_hours((load.pickup_x, load.pickup_y), (load.dropoff_x, load.dropoff_y))
                + load.service_hours
            )
            if needed <= truck.available_hours:
                actions.append(
                    ControlAction(
                        kind="reassign",
                        load_id=load.id,
                        target_truck_id=truck.id,
                        label=f"Reassign {load.id} from {current.id} to {truck.id}",
                    )
                )
            if len(actions) >= max_candidates:
                return actions

    return actions


def apply_action(state: FleetState, action: ControlAction) -> FleetState:
    if action.kind == "none":
        return state
    if action.kind == "reserve" and action.load_id:
        load = state.load(action.load_id)
        reserve_id = f"R-{load.id}"
        reserve = Truck(
            id=reserve_id,
            x=load.pickup_x,
            y=load.pickup_y,
            available_hours=8.0,
            capacity=1.0,
            active=True,
        )
        updated_load = Load(
            id=load.id,
            truck_id=reserve_id,
            pickup_x=load.pickup_x,
            pickup_y=load.pickup_y,
            dropoff_x=load.dropoff_x,
            dropoff_y=load.dropoff_y,
            due_in_hours=load.due_in_hours,
            service_hours=load.service_hours,
            priority=load.priority,
        )
        return FleetState(
            time_hours=state.time_hours,
            trucks=state.trucks + (reserve,),
            loads=tuple(updated_load if l.id == load.id else l for l in state.loads),
        )
    if action.kind == "reassign" and action.load_id and action.target_truck_id:
        load = state.load(action.load_id)
        updated = Load(
            id=load.id,
            truck_id=action.target_truck_id,
            pickup_x=load.pickup_x,
            pickup_y=load.pickup_y,
            dropoff_x=load.dropoff_x,
            dropoff_y=load.dropoff_y,
            due_in_hours=load.due_in_hours,
            service_hours=load.service_hours,
            priority=load.priority,
        )
        return state.replace_load(updated)
    return state


def rank_controls(state: FleetState, rollouts: int = 160, seed: int = 11) -> list[RankedAction]:
    baseline = evaluate_pcc(state, rollouts=rollouts, seed=seed)
    ranked: list[RankedAction] = []

    for i, action in enumerate(candidate_actions(state)):
        candidate_state = apply_action(state, action)
        metrics = evaluate_pcc(candidate_state, rollouts=rollouts, seed=seed + i)
        improvement = baseline.instability - metrics.instability
        ranked.append(RankedAction(action, metrics, improvement))

    ranked.sort(key=lambda r: (r.metrics.instability, r.metrics.cascade_risk))
    return ranked
