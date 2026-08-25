from roadstar_sentinel.experiments import generate_specs, run_experiment, simulate_spec


def test_generated_experiment_contains_multiple_families():
    specs = generate_specs(n=40, seed=12)
    assert len(specs) == 40
    assert len({s.family for s in specs}) >= 3


def test_batch_experiment_scores_all_detectors():
    summary, results = run_experiment(n=30, seed=14, rollouts=30)
    assert len(results) == 30
    assert summary.failures + summary.non_failures == 30
    assert len(summary.detectors) == 4
    for detector in summary.detectors:
        assert 0.0 <= detector.precision <= 1.0
        assert 0.0 <= detector.recall <= 1.0
        assert 0.0 <= detector.false_positive_rate <= 1.0


def test_stable_spec_can_remain_nonfailure():
    spec = next(s for s in generate_specs(n=100, seed=99) if s.family == "stable")
    result = simulate_spec(spec, duration_hours=2.0, rollouts=25)
    assert result.spec.family == "stable"
