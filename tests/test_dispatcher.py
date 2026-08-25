from roadstar_sentinel.closed_loop import run_closed_loop
from roadstar_sentinel.dispatcher import answer_dispatcher_question, build_dispatcher_brief, compare_outcomes
from roadstar_sentinel.scenarios import creeping_failure_scenario


def _result():
    return run_closed_loop(creeping_failure_scenario(), rollouts=40, seed=23)


def test_dispatcher_brief_is_grounded_and_actionable():
    result = _result()
    brief = build_dispatcher_brief(result)
    assert brief.status == "ACTION RECOMMENDED"
    assert result.selected.action.label in brief.headline
    assert len(brief.causal_chain) == 4
    assert "CPU" in brief.model_basis


def test_dispatcher_question_routes_why_and_no_action():
    result = _result()
    why = answer_dispatcher_question("Why are you recommending this now?", result)
    no_action = answer_dispatcher_question("What happens if I do nothing?", result)
    assert why["intent"] == "explain_warning"
    assert no_action["intent"] == "no_action_counterfactual"
    assert why["grounded"] is True


def test_compare_outcomes_returns_winner():
    result = _result()
    options = [x for x in result.alternatives if x.action.kind != "none"]
    assert len(options) >= 2
    compared = compare_outcomes(options[0], options[1])
    assert compared["winner"] in {options[0].action.label, options[1].action.label}
