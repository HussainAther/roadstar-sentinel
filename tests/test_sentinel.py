from roadstar_sentinel.controls import rank_controls
from roadstar_sentinel.pcc import evaluate_pcc
from roadstar_sentinel.scenarios import baseline_scenario, disrupted_scenario


def test_breakdown_increases_pressure_or_instability():
    base = evaluate_pcc(baseline_scenario(), rollouts=120, seed=1)
    disrupted = evaluate_pcc(disrupted_scenario(), rollouts=120, seed=1)
    assert disrupted.pressure >= base.pressure
    assert disrupted.instability >= base.instability


def test_controller_produces_ranked_actions():
    ranked = rank_controls(disrupted_scenario(), rollouts=80, seed=2)
    assert ranked
    assert any(r.action.kind == "reassign" for r in ranked)
