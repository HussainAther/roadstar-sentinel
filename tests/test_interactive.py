from roadstar_sentinel.interactive import analyze_incident, incident_catalog, simulate_incident


def test_incident_catalog_has_expected_demo_scenarios():
    kinds = {x["kind"] for x in incident_catalog()}
    assert {"truck_breakdown", "traffic_surge", "hos_shortage", "depot_outage", "compound_shock"} <= kinds


def test_all_interactive_incidents_generate_cpu_trajectories():
    for item in incident_catalog():
        points = simulate_incident(item["kind"], severity=0.7, duration_hours=2.0, rollouts=25, seed=9)
        assert len(points) == 9
        assert points[0].time_hours == 0.0
        assert all(0.0 <= p.metrics.instability <= 1.0 for p in points)


def test_compound_shock_returns_ranked_control_and_counterfactual():
    result = analyze_incident("compound_shock", severity=0.8, rollouts=30, seed=11)
    assert result["selected_action"]["kind"] in {"none", "reserve", "reassign"}
    assert result["recommendations"]
    assert len(result["no_control"]) == len(result["sentinel_control"])
    assert result["selected_action"]["improvement_vs_no_action"] >= 0.0


def test_operator_can_simulate_non_default_action():
    base = analyze_incident("compound_shock", severity=0.8, rollouts=30, seed=11)
    options = [x for x in base["recommendations"] if x["kind"] != "none"]
    assert options
    from roadstar_sentinel.interactive import analyze_selected_action
    result = analyze_selected_action("compound_shock", options[-1]["action_id"], severity=0.8, rollouts=30, seed=11)
    assert "error" not in result
    assert result["chosen_action"]["action_id"] == options[-1]["action_id"]
    assert len(result["chosen_control"]) == len(result["no_control"])
    assert result["regret_vs_sentinel"] >= -1e-9


def test_invalid_operator_action_is_rejected_cleanly():
    from roadstar_sentinel.interactive import analyze_selected_action
    result = analyze_selected_action("traffic_surge", "not:a:real:action", severity=0.7, rollouts=20, seed=5)
    assert "error" in result
    assert result["valid_actions"]
