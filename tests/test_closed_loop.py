from roadstar_sentinel.closed_loop import run_closed_loop
from roadstar_sentinel.scenarios import creeping_failure_scenario


def test_closed_loop_branches_at_warning_and_selects_action():
    result = run_closed_loop(creeping_failure_scenario(), rollouts=50, seed=23)
    assert result.control_time_hours > 0
    assert result.selected.action.kind in {"none", "reassign", "reserve"}
    assert len(result.no_control) == len(result.sentinel_control)
    assert result.alternatives


def test_closed_loop_improves_projected_system_cost():
    result = run_closed_loop(creeping_failure_scenario(), rollouts=70, seed=23)
    no_action = next(a for a in result.alternatives if a.action.kind == "none")
    assert result.selected.objective <= no_action.objective
    assert result.instability_auc_control <= result.instability_auc_no_control + 1e-9
