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
