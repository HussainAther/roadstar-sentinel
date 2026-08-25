from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, median

from .dynamics import sentinel_alarm
from .models import FleetState, Load, Truck
from .pcc import evaluate_pcc
from .scenarios import creeping_failure_scenario


@dataclass(frozen=True)
class DisturbanceSpec:
    family: str
    target_truck_id: str
    severity: float
    onset_hours: float
    failure_time_hours: float | None
    seed: int


@dataclass(frozen=True)
class ExperimentPoint:
    time_hours: float
    pressure: float
    entropy: float
    control: float
    instability: float
    cascade_risk: float
    entropy_rate: float
    sentinel_alarm: bool
    pressure_alarm: bool
    composite_alarm: bool
    conventional_alarm: bool


@dataclass(frozen=True)
class ScenarioResult:
    spec: DisturbanceSpec
    failed: bool
    failure_time_hours: float | None
    points: tuple[ExperimentPoint, ...]


@dataclass(frozen=True)
class DetectorMetrics:
    name: str
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    false_positive_rate: float
    mean_lead_time_hours: float | None
    median_lead_time_hours: float | None


@dataclass(frozen=True)
class ExperimentSummary:
    scenarios: int
    failures: int
    non_failures: int
    family_counts: dict[str, int]
    detectors: tuple[DetectorMetrics, ...]
    sentinel_lead_times_hours: tuple[float, ...]


def _clamp_nonnegative(x: float) -> float:
    return max(0.0, x)


def _state_at(base: FleetState, spec: DisturbanceSpec, t: float) -> FleetState:
    """Evolve a synthetic fleet under one disturbance family.

    This is deliberately lightweight and CPU-friendly. It is a scenario generator,
    not a calibrated model of any specific carrier or road network.
    """
    elapsed = max(0.0, t - spec.onset_hours)
    sev = spec.severity
    if spec.failure_time_hours is None:
        progress = min(1.0, elapsed / 4.0)
    else:
        span = max(spec.failure_time_hours - spec.onset_hours, 0.25)
        progress = min(1.0, elapsed / span)

    trucks: list[Truck] = []
    for truck in base.trucks:
        available = _clamp_nonnegative(truck.available_hours - t)
        active = truck.active

        if truck.id == spec.target_truck_id and elapsed > 0:
            if spec.family == "mechanical":
                available = _clamp_nonnegative(available - sev * (0.65 * elapsed + 5.5 * progress ** 2))
            elif spec.family == "hos":
                available = _clamp_nonnegative(available - sev * (0.95 * elapsed + 4.8 * progress ** 1.5))
            elif spec.family == "compound":
                available = _clamp_nonnegative(available - sev * (0.90 * elapsed + 6.5 * progress ** 2))
            elif spec.family == "stable":
                available = _clamp_nonnegative(available - sev * 0.12 * elapsed)

            if spec.failure_time_hours is not None and t >= spec.failure_time_hours:
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

    loads: list[Load] = []
    for load in base.loads:
        due = _clamp_nonnegative(load.due_in_hours - t)
        if elapsed > 0:
            if spec.family == "congestion":
                due -= sev * (0.35 * elapsed + 4.5 * progress ** 2)
            elif spec.family == "compound":
                due -= sev * (0.30 * elapsed + 3.5 * progress ** 2)
            elif spec.family == "stable":
                due -= sev * 0.04 * elapsed
        loads.append(
            Load(
                id=load.id,
                truck_id=load.truck_id,
                pickup_x=load.pickup_x,
                pickup_y=load.pickup_y,
                dropoff_x=load.dropoff_x,
                dropoff_y=load.dropoff_y,
                due_in_hours=max(0.1, due),
                service_hours=load.service_hours,
                priority=load.priority,
            )
        )

    return FleetState(time_hours=t, trucks=tuple(trucks), loads=tuple(loads))


def _conventional_alarm(state: FleetState) -> bool:
    # Conventional alarm in the experiment = a hard operational constraint has
    # already tripped: truck inactive or assigned truck has no usable hours left.
    for load in state.loads:
        truck = state.truck(load.truck_id)
        if not truck.active or truck.available_hours <= 0.05:
            return True
    return False


def _experiment_base() -> FleetState:
    base = creeping_failure_scenario()
    trucks = tuple(
        Truck(t.id, t.x, t.y, t.available_hours + 5.0, t.capacity, t.active)
        for t in base.trucks
    )
    loads = tuple(
        Load(
            l.id, l.truck_id, l.pickup_x, l.pickup_y, l.dropoff_x, l.dropoff_y,
            l.due_in_hours + 5.0, l.service_hours, l.priority
        )
        for l in base.loads
    )
    return FleetState(time_hours=0.0, trucks=trucks, loads=loads)



def sentinel_composite_score(point_metrics, entropy_rate: float) -> float:
    """Scalar PCC operating score used for ablation/threshold experiments."""
    rate = max(0.0, min(1.0, entropy_rate / 0.7))
    return (
        0.30 * point_metrics.instability
        + 0.22 * point_metrics.entropy
        + 0.18 * point_metrics.pressure
        + 0.15 * point_metrics.cascade_risk
        + 0.15 * rate
    )

def simulate_spec(
    spec: DisturbanceSpec,
    duration_hours: float = 3.0,
    step_hours: float = 0.25,
    rollouts: int = 70,
) -> ScenarioResult:
    base = _experiment_base()
    raw: list[tuple[float, FleetState, object]] = []
    steps = int(round(duration_hours / step_hours))

    for i in range(steps + 1):
        t = round(i * step_hours, 10)
        state = _state_at(base, spec, t)
        metrics = evaluate_pcc(state, rollouts=rollouts, seed=spec.seed * 1000 + i)
        raw.append((t, state, metrics))

    points: list[ExperimentPoint] = []
    for i, (t, state, metrics) in enumerate(raw):
        # Use a short rolling slope rather than a one-step derivative so the
        # detector is less sensitive to Monte Carlo jitter on a laptop-scale run.
        if i < 3:
            entropy_rate = 0.0
        else:
            prev_t, _, prev = raw[i - 3]
            entropy_rate = (metrics.entropy - prev.entropy) / max(t - prev_t, 1e-9)

        conventional = _conventional_alarm(state)
        points.append(
            ExperimentPoint(
                time_hours=t,
                pressure=metrics.pressure,
                entropy=metrics.entropy,
                control=metrics.control,
                instability=metrics.instability,
                cascade_risk=metrics.cascade_risk,
                entropy_rate=entropy_rate,
                sentinel_alarm=sentinel_alarm(metrics, entropy_rate),
                pressure_alarm=metrics.pressure >= 0.42,
                composite_alarm=sentinel_composite_score(metrics, entropy_rate) >= 0.29,
                conventional_alarm=conventional,
            )
        )

    failure_point = next((p for p in points if p.conventional_alarm), None)
    return ScenarioResult(
        spec=spec,
        failed=failure_point is not None,
        failure_time_hours=failure_point.time_hours if failure_point else None,
        points=tuple(points),
    )


def generate_specs(n: int = 120, seed: int = 2026) -> list[DisturbanceSpec]:
    rng = random.Random(seed)
    trucks = ["T01", "T02", "T03", "T04", "T05", "T06"]
    families = ["stable", "mechanical", "hos", "congestion", "compound"]
    weights = [0.28, 0.22, 0.18, 0.16, 0.16]
    specs: list[DisturbanceSpec] = []

    for i in range(n):
        family = rng.choices(families, weights=weights, k=1)[0]
        target = rng.choice(trucks)
        onset = rng.choice([0.0, 0.25, 0.5, 0.75])

        if family == "stable":
            severity = rng.uniform(0.05, 0.28)
            failure_time = None
        elif family == "congestion":
            severity = rng.uniform(0.55, 1.2)
            # Some congestion episodes clear; severe ones culminate in a hard constraint.
            failure_time = rng.uniform(2.0, 2.75) if severity > 0.92 else None
        elif family == "mechanical":
            severity = rng.uniform(0.75, 1.35)
            failure_time = rng.uniform(1.75, 2.75)
        elif family == "hos":
            severity = rng.uniform(0.75, 1.35)
            failure_time = rng.uniform(2.0, 2.75)
        else:  # compound
            severity = rng.uniform(0.85, 1.45)
            failure_time = rng.uniform(1.5, 2.5)

        specs.append(
            DisturbanceSpec(
                family=family,
                target_truck_id=target,
                severity=severity,
                onset_hours=onset,
                failure_time_hours=failure_time,
                seed=seed + i + 1,
            )
        )
    return specs


def _first_alarm(result: ScenarioResult, detector: str) -> float | None:
    for point in result.points:
        flag = {
            "Sentinel PCC+entropy": point.sentinel_alarm,
            "PCC composite": point.composite_alarm,
            "Pressure-only": point.pressure_alarm,
            "Conventional hard alarm": point.conventional_alarm,
        }[detector]
        if flag:
            return point.time_hours
    return None


def _score_detector(results: list[ScenarioResult], detector: str) -> DetectorMetrics:
    tp = fp = tn = fn = 0
    leads: list[float] = []

    for result in results:
        alarm_t = _first_alarm(result, detector)
        if result.failed:
            if alarm_t is not None and result.failure_time_hours is not None and alarm_t <= result.failure_time_hours:
                tp += 1
                leads.append(max(0.0, result.failure_time_hours - alarm_t))
            else:
                fn += 1
        else:
            if alarm_t is None:
                tn += 1
            else:
                fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return DetectorMetrics(
        name=detector,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        mean_lead_time_hours=mean(leads) if leads else None,
        median_lead_time_hours=median(leads) if leads else None,
    )


def run_experiment(n: int = 120, seed: int = 2026, rollouts: int = 70) -> tuple[ExperimentSummary, list[ScenarioResult]]:
    specs = generate_specs(n=n, seed=seed)
    results = [simulate_spec(spec, rollouts=rollouts) for spec in specs]
    family_counts: dict[str, int] = {}
    for result in results:
        family_counts[result.spec.family] = family_counts.get(result.spec.family, 0) + 1

    detectors = tuple(
        _score_detector(results, name)
        for name in ("Sentinel PCC+entropy", "PCC composite", "Pressure-only", "Conventional hard alarm")
    )
    sentinel_leads = []
    for result in results:
        if not result.failed or result.failure_time_hours is None:
            continue
        alarm_t = _first_alarm(result, "Sentinel PCC+entropy")
        if alarm_t is not None and alarm_t <= result.failure_time_hours:
            sentinel_leads.append(result.failure_time_hours - alarm_t)

    summary = ExperimentSummary(
        scenarios=len(results),
        failures=sum(r.failed for r in results),
        non_failures=sum(not r.failed for r in results),
        family_counts=family_counts,
        detectors=detectors,
        sentinel_lead_times_hours=tuple(sentinel_leads),
    )
    return summary, results


def percentile(values: tuple[float, ...], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
