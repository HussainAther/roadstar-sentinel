from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .controls import apply_action, candidate_actions
from .dynamics import TrajectoryPoint, degraded_state, deterministic_failure, sentinel_alarm, simulate_trajectory
from .models import ControlAction, FleetState
from .pcc import evaluate_pcc


@dataclass(frozen=True)
class ControlOutcome:
    action: ControlAction
    objective: float
    mean_instability: float
    mean_cascade_risk: float
    peak_instability: float
    failure_time_hours: float | None
    failure_avoided: bool
    failure_delay_hours: float | None


@dataclass(frozen=True)
class ClosedLoopComparison:
    control_time_hours: float
    selected: ControlOutcome
    alternatives: tuple[ControlOutcome, ...]
    no_control: tuple[TrajectoryPoint, ...]
    sentinel_control: tuple[TrajectoryPoint, ...]
    instability_auc_no_control: float
    instability_auc_control: float
    cascade_auc_no_control: float
    cascade_auc_control: float

    @property
    def instability_auc_reduction(self) -> float:
        if self.instability_auc_no_control <= 0:
            return 0.0
        return max(0.0, 1.0 - self.instability_auc_control / self.instability_auc_no_control)

    @property
    def cascade_auc_reduction(self) -> float:
        if self.cascade_auc_no_control <= 0:
            return 0.0
        return max(0.0, 1.0 - self.cascade_auc_control / self.cascade_auc_no_control)


def _first_failure(points: list[TrajectoryPoint] | tuple[TrajectoryPoint, ...]) -> float | None:
    return next((p.time_hours for p in points if p.conventional_failure), None)


def _auc(points: list[TrajectoryPoint] | tuple[TrajectoryPoint, ...], key: str, start_time: float) -> float:
    relevant = [p for p in points if p.time_hours >= start_time]
    if len(relevant) < 2:
        return 0.0
    area = 0.0
    for a, b in zip(relevant, relevant[1:]):
        ya = getattr(a.metrics, key)
        yb = getattr(b.metrics, key)
        area += 0.5 * (ya + yb) * (b.time_hours - a.time_hours)
    return area


def _trajectory_with_action(
    base: FleetState,
    no_control: list[TrajectoryPoint],
    action: ControlAction,
    control_time: float,
    rollouts: int,
    seed: int,
) -> list[TrajectoryPoint]:
    """Replay the same disturbance, applying one selected action at warning time.

    The control action is persisted in the base assignment for all post-warning
    states. Pre-warning points are copied from the no-control run, so the
    counterfactual branches only when Sentinel actually intervenes.
    """
    controlled_base = apply_action(base, action)
    points: list[TrajectoryPoint] = []
    previous_entropy: float | None = None
    previous_pressure: float | None = None
    previous_time: float | None = None

    for i, original in enumerate(no_control):
        t = original.time_hours
        if t < control_time:
            points.append(original)
            previous_entropy = original.metrics.entropy
            previous_pressure = original.metrics.pressure
            previous_time = t
            continue

        state = degraded_state(controlled_base, t)
        metrics = evaluate_pcc(state, rollouts=rollouts, seed=seed + i)
        if previous_time is None or previous_entropy is None or previous_pressure is None:
            entropy_rate = 0.0
            pressure_rate = 0.0
        else:
            dt = max(t - previous_time, 1e-9)
            entropy_rate = (metrics.entropy - previous_entropy) / dt
            pressure_rate = (metrics.pressure - previous_pressure) / dt

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
        previous_entropy = metrics.entropy
        previous_pressure = metrics.pressure
        previous_time = t

    return points


def _score_action(
    base: FleetState,
    no_control: list[TrajectoryPoint],
    action: ControlAction,
    control_time: float,
    rollouts: int,
    seed: int,
) -> tuple[ControlOutcome, list[TrajectoryPoint]]:
    controlled = _trajectory_with_action(base, no_control, action, control_time, rollouts, seed)
    post = [p for p in controlled if p.time_hours >= control_time]
    mean_instability = mean(p.metrics.instability for p in post)
    mean_cascade = mean(p.metrics.cascade_risk for p in post)
    peak_instability = max(p.metrics.instability for p in post)
    failure_time = _first_failure(controlled)
    baseline_failure = _first_failure(no_control)

    if baseline_failure is not None and failure_time is None:
        failure_avoided = True
        failure_delay = None
    else:
        failure_avoided = False
        failure_delay = (
            failure_time - baseline_failure
            if baseline_failure is not None and failure_time is not None and failure_time > baseline_failure
            else None
        )

    failure_fraction = sum(p.conventional_failure for p in post) / max(len(post), 1)
    action_penalty = 0.0 if action.kind == "none" else 0.015
    objective = mean_instability + 0.35 * mean_cascade + 0.25 * failure_fraction + action_penalty

    return (
        ControlOutcome(
            action=action,
            objective=objective,
            mean_instability=mean_instability,
            mean_cascade_risk=mean_cascade,
            peak_instability=peak_instability,
            failure_time_hours=failure_time,
            failure_avoided=failure_avoided,
            failure_delay_hours=failure_delay,
        ),
        controlled,
    )


def run_closed_loop(
    base: FleetState,
    duration_hours: float = 3.0,
    step_hours: float = 0.25,
    rollouts: int = 120,
    seed: int = 23,
    max_candidates: int = 18,
) -> ClosedLoopComparison:
    """Detect, intervene, and re-simulate a synthetic PCC feedback loop on CPU."""
    no_control = simulate_trajectory(
        base,
        duration_hours=duration_hours,
        step_hours=step_hours,
        rollouts=rollouts,
        seed=seed,
    )
    warning_point = next((p for p in no_control if p.early_warning), None)
    if warning_point is None:
        raise RuntimeError("No Sentinel warning occurred; cannot branch the feedback demo.")

    warning_state = degraded_state(base, warning_point.time_hours)
    actions = candidate_actions(warning_state, max_candidates=max_candidates)

    scored: list[tuple[ControlOutcome, list[TrajectoryPoint]]] = []
    for action in actions:
        # Paired random seeds keep action comparisons as fair as possible.
        scored.append(
            _score_action(
                base,
                no_control,
                action,
                warning_point.time_hours,
                rollouts,
                seed,
            )
        )

    scored.sort(key=lambda item: (item[0].objective, item[0].peak_instability))
    selected, controlled = scored[0]
    alternatives = tuple(item[0] for item in scored[:6])

    return ClosedLoopComparison(
        control_time_hours=warning_point.time_hours,
        selected=selected,
        alternatives=alternatives,
        no_control=tuple(no_control),
        sentinel_control=tuple(controlled),
        instability_auc_no_control=_auc(no_control, "instability", warning_point.time_hours),
        instability_auc_control=_auc(controlled, "instability", warning_point.time_hours),
        cascade_auc_no_control=_auc(no_control, "cascade_risk", warning_point.time_hours),
        cascade_auc_control=_auc(controlled, "cascade_risk", warning_point.time_hours),
    )
