from __future__ import annotations

from .controls import rank_controls
from .closed_loop import run_closed_loop
from .experiments import run_experiment
from .dynamics import simulate_trajectory, warning_lead_time
from .pcc import evaluate_pcc
from .scenarios import baseline_scenario, creeping_failure_scenario, disrupted_scenario


def fmt(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    base = baseline_scenario()
    disrupted = disrupted_scenario()

    print("ROADSTAR SENTINEL v0.4")
    print("Pressure -> Chaos -> Control")
    print()

    for label, state in [("BASELINE", base), ("DISRUPTED: T03 BREAKDOWN", disrupted)]:
        m = evaluate_pcc(state)
        print(label)
        print(f"  Pressure      {fmt(m.pressure)}")
        print(f"  Entropy       {fmt(m.entropy)}")
        print(f"  Control       {fmt(m.control)}")
        print(f"  Instability   {fmt(m.instability)}")
        print(f"  Cascade risk  {fmt(m.cascade_risk)}")
        print()

    print("DYNAMIC EARLY-WARNING RUN: CREEPING T03 FAILURE")
    trajectory = simulate_trajectory(creeping_failure_scenario())
    lead = warning_lead_time(trajectory)
    for point in trajectory:
        marker = ""
        if point.early_warning:
            marker += " WARNING"
        if point.conventional_failure:
            marker += " FAILURE"
        print(
            f"  t={point.time_hours:>4.2f}h  P={fmt(point.metrics.pressure)} "
            f"H={fmt(point.metrics.entropy)} dH/dt={point.entropy_rate:+.3f} "
            f"C={fmt(point.metrics.control)} I={fmt(point.metrics.instability)}{marker}"
        )
    print()
    if lead is None:
        print("  Sentinel did not produce a pre-failure lead time in this run.")
    else:
        print(f"  Sentinel warning lead time: {lead:.2f} hours")
    print()

    print("TOP CONTROL RECOMMENDATIONS")
    for rank, result in enumerate(rank_controls(disrupted)[:5], start=1):
        print(
            f"{rank}. {result.action.label} | "
            f"instability={fmt(result.metrics.instability)} | "
            f"cascade={fmt(result.metrics.cascade_risk)} | "
            f"improvement={fmt(result.improvement)}"
        )

    print()
    print("CPU BATCH EXPERIMENT (120 SYNTHETIC SCENARIOS)")
    summary, _ = run_experiment(n=120, seed=2026, rollouts=80)
    print(f"  scenarios={summary.scenarios} failures={summary.failures} non_failures={summary.non_failures}")
    for d in summary.detectors:
        lead = "n/a" if d.median_lead_time_hours is None else f"{d.median_lead_time_hours:.2f}h"
        print(
            f"  {d.name:<24} precision={d.precision:.3f} recall={d.recall:.3f} "
            f"FPR={d.false_positive_rate:.3f} median_lead={lead}"
        )

    print()
    print("CLOSED-LOOP SENTINEL CONTROL")
    loop = run_closed_loop(creeping_failure_scenario(), rollouts=100, seed=23)
    print(f"  control time: {loop.control_time_hours:.2f}h")
    print(f"  selected: {loop.selected.action.label}")
    print(f"  instability burden reduction: {100*loop.instability_auc_reduction:.1f}%")
    print(f"  cascade burden reduction: {100*loop.cascade_auc_reduction:.1f}%")
    if loop.selected.failure_avoided:
        print("  conventional failure: avoided in simulated horizon")
    elif loop.selected.failure_delay_hours is not None:
        print(f"  conventional failure delayed: {loop.selected.failure_delay_hours:.2f}h")
    else:
        print("  conventional failure: not avoided in simulated horizon")


if __name__ == "__main__":
    main()
