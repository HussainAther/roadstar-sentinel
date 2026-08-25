from __future__ import annotations

from dataclasses import asdict, dataclass

from .closed_loop import ClosedLoopComparison, ControlOutcome


@dataclass(frozen=True)
class CausalFactor:
    name: str
    severity: str
    evidence: str
    implication: str


@dataclass(frozen=True)
class DispatcherBrief:
    status: str
    headline: str
    summary: str
    why_now: str
    causal_chain: tuple[CausalFactor, ...]
    recommended_action: str
    expected_effect: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    confidence: str
    confidence_reason: str
    no_action_consequence: str
    model_basis: str

    def as_dict(self) -> dict:
        data = asdict(self)
        data["causal_chain"] = [asdict(x) for x in self.causal_chain]
        return data


def _at_control(result: ClosedLoopComparison):
    return min(result.no_control, key=lambda p: abs(p.time_hours - result.control_time_hours))


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _severity(value: float, medium: float = 0.45, high: float = 0.65) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "elevated"
    return "moderate"


def _confidence(result: ClosedLoopComparison) -> tuple[str, str]:
    options = sorted(result.alternatives, key=lambda a: a.objective)
    if len(options) < 2:
        return "moderate", "Only one feasible intervention was available in this synthetic state."
    best, second = options[0], options[1]
    margin = (second.objective - best.objective) / max(abs(second.objective), 1e-9)
    if margin >= 0.08:
        return "high", f"The selected action has a {100 * margin:.1f}% objective margin over the next-best candidate."
    if margin >= 0.025:
        return "moderate", f"The selected action has a {100 * margin:.1f}% objective margin over the next-best candidate."
    return "low", f"The top alternatives are close: only a {100 * margin:.1f}% objective margin separates the best two."


def build_dispatcher_brief(result: ClosedLoopComparison) -> DispatcherBrief:
    point = _at_control(result)
    m = point.metrics
    baseline_failure = next((p.time_hours for p in result.no_control if p.conventional_failure), None)
    confidence, confidence_reason = _confidence(result)

    factors = (
        CausalFactor(
            name="Operational pressure",
            severity=_severity(m.pressure),
            evidence=f"Pressure is {_pct(m.pressure)} and changing at {point.pressure_rate:+.2f}/h.",
            implication="Schedule and resource slack are shrinking, so ordinary delays have less room to dissipate.",
        ),
        CausalFactor(
            name="Predictive entropy",
            severity=_severity(m.entropy, 0.50, 0.66),
            evidence=f"Future-state entropy is {_pct(m.entropy)} with dH/dt={point.entropy_rate:+.2f}/h.",
            implication="Forecast rollouts are diverging; the fleet has more materially different ways to fail downstream.",
        ),
        CausalFactor(
            name="Control authority",
            severity="high" if m.control < 0.45 else "elevated" if m.control < 0.70 else "moderate",
            evidence=f"Remaining control authority is {_pct(m.control)}.",
            implication="Fewer low-cost recovery actions remain available as the disturbance progresses.",
        ),
        CausalFactor(
            name="Cascade exposure",
            severity=_severity(m.cascade_risk, 0.45, 0.70),
            evidence=f"Simulated cascade risk is {_pct(m.cascade_risk)} at the warning point.",
            implication="A local disruption is increasingly likely to propagate into later assignments and service constraints.",
        ),
    )

    effect = [
        f"Reduce post-warning instability burden by {_pct(result.instability_auc_reduction)} versus no control.",
        f"Reduce post-warning cascade-risk burden by {_pct(result.cascade_auc_reduction)} versus no control.",
        f"Projected mean instability after intervention: {_pct(result.selected.mean_instability)}.",
    ]
    if result.selected.failure_avoided:
        effect.append("Avoid the synthetic conventional-failure condition within the modeled horizon.")
    elif result.selected.failure_delay_hours is not None:
        effect.append(f"Delay the synthetic failure condition by {result.selected.failure_delay_hours:.2f} h.")
    else:
        effect.append("Improve the trajectory, but not fully avoid the modeled failure in this scenario.")

    tradeoffs = (
        "The controller optimizes a prototype systems cost, not a production dispatch objective such as dollars or contractual SLA penalties.",
        "Reserve/reassignment actions may carry operational cost that is represented only by a simple action penalty in this MVP.",
        "Recommendations are counterfactual simulation results on synthetic data and require dispatcher judgment.",
    )

    failure_text = (
        f"Without intervention, the current seeded scenario reaches the conventional failure condition at t={baseline_failure:.2f} h."
        if baseline_failure is not None
        else "No conventional hard failure occurs within the modeled no-control horizon, but instability remains elevated."
    )

    return DispatcherBrief(
        status="ACTION RECOMMENDED",
        headline=f"Intervene now: {result.selected.action.label}",
        summary=(
            f"Sentinel detected a rising-instability regime at t={result.control_time_hours:.2f} h. "
            f"Pressure, future-state uncertainty, and cascade exposure are jointly increasing before the hard failure threshold."
        ),
        why_now=(
            f"The warning is driven by the combination of {_pct(m.pressure)} pressure, {_pct(m.entropy)} predictive entropy, "
            f"and dH/dt={point.entropy_rate:+.2f}/h rather than by a single KPI threshold."
        ),
        causal_chain=factors,
        recommended_action=result.selected.action.label,
        expected_effect=tuple(effect),
        tradeoffs=tradeoffs,
        confidence=confidence,
        confidence_reason=confidence_reason,
        no_action_consequence=failure_text,
        model_basis=(
            "Grounded in the local PCC state, CPU Monte Carlo future rollouts, and paired counterfactual evaluation of feasible control actions. "
            "No cloud LLM or GPU is required for this explanation."
        ),
    )


def compare_outcomes(a: ControlOutcome, b: ControlOutcome) -> dict:
    winner = a if a.objective <= b.objective else b
    loser = b if winner is a else a
    return {
        "winner": winner.action.label,
        "summary": (
            f"{winner.action.label} ranks ahead of {loser.action.label} because its projected systems objective is "
            f"{winner.objective:.3f} versus {loser.objective:.3f}."
        ),
        "differences": {
            "objective": a.objective - b.objective,
            "mean_instability": a.mean_instability - b.mean_instability,
            "mean_cascade_risk": a.mean_cascade_risk - b.mean_cascade_risk,
            "peak_instability": a.peak_instability - b.peak_instability,
        },
        "a": {
            "action": a.action.label,
            "objective": a.objective,
            "mean_instability": a.mean_instability,
            "mean_cascade_risk": a.mean_cascade_risk,
            "peak_instability": a.peak_instability,
        },
        "b": {
            "action": b.action.label,
            "objective": b.objective,
            "mean_instability": b.mean_instability,
            "mean_cascade_risk": b.mean_cascade_risk,
            "peak_instability": b.peak_instability,
        },
    }


def answer_dispatcher_question(question: str, result: ClosedLoopComparison) -> dict:
    """CPU-only grounded dispatcher copilot for the demo.

    This intentionally uses transparent intent routing over computed Sentinel state.
    A production version can place an LLM over the same structured evidence without
    changing the control engine or requiring one for core functionality.
    """
    q = question.strip().lower()
    brief = build_dispatcher_brief(result)
    point = _at_control(result)

    if any(x in q for x in ("why", "reason", "cause")):
        answer = brief.why_now + " " + " ".join(f"{x.name}: {x.implication}" for x in brief.causal_chain[:3])
        intent = "explain_warning"
    elif any(x in q for x in ("no action", "do nothing", "ignore", "without")):
        answer = brief.no_action_consequence + " " + brief.expected_effect[0]
        intent = "no_action_counterfactual"
    elif any(x in q for x in ("confidence", "sure", "certain")):
        answer = f"Recommendation confidence is {brief.confidence}. {brief.confidence_reason}"
        intent = "confidence"
    elif any(x in q for x in ("tradeoff", "downside", "cost", "risk of")):
        answer = " ".join(brief.tradeoffs)
        intent = "tradeoffs"
    elif any(x in q for x in ("compare", "alternative", "instead")):
        options = [x for x in result.alternatives if x.action.kind != "none"]
        if len(options) >= 2:
            comparison = compare_outcomes(options[0], options[1])
            answer = comparison["summary"]
        else:
            comparison = None
            answer = "There are not enough feasible alternatives in the current state for a meaningful comparison."
        return {"intent": "compare_actions", "answer": answer, "comparison": comparison, "grounded": True}
    elif any(x in q for x in ("status", "happening", "state", "risk")):
        answer = (
            f"At t={result.control_time_hours:.2f} h the fleet is in an early-warning regime: pressure {_pct(point.metrics.pressure)}, "
            f"entropy {_pct(point.metrics.entropy)}, control authority {_pct(point.metrics.control)}, and cascade risk {_pct(point.metrics.cascade_risk)}. "
            f"{brief.no_action_consequence}"
        )
        intent = "fleet_status"
    elif any(x in q for x in ("recommend", "what should", "do now", "action")):
        answer = f"Recommended action: {brief.recommended_action}. {brief.expected_effect[0]} {brief.confidence_reason}"
        intent = "recommend_action"
    else:
        answer = (
            f"{brief.headline}. {brief.summary} {brief.expected_effect[0]} "
            "Ask why the warning fired, what happens with no action, compare alternatives, or ask about confidence/tradeoffs."
        )
        intent = "dispatcher_summary"

    return {"intent": intent, "answer": answer, "grounded": True}
