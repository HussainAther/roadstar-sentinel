from roadstar_sentinel.dynamics import simulate_trajectory, warning_lead_time
from roadstar_sentinel.scenarios import baseline_scenario, creeping_failure_scenario


def test_dynamic_run_has_time_series_and_rates():
    points = simulate_trajectory(baseline_scenario(), duration_hours=2.5, step_hours=0.25, rollouts=90, seed=10)
    assert len(points) == 11
    assert points[0].entropy_rate == 0.0
    assert any(abs(p.entropy_rate) > 0 for p in points[1:])


def test_sentinel_can_warn_before_conventional_failure():
    points = simulate_trajectory(creeping_failure_scenario(), duration_hours=3.0, step_hours=0.25, rollouts=160, seed=23)
    warning = next((p.time_hours for p in points if p.early_warning), None)
    failure = next((p.time_hours for p in points if p.conventional_failure), None)
    assert warning is not None
    assert failure is not None
    assert warning < failure
    assert warning_lead_time(points) == failure - warning
