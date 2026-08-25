from roadstar_sentinel.operations import operations_view


def test_operations_view_contains_fleet_and_timeline():
    view = operations_view("compound_shock", severity=0.8, time_hours=1.0)
    assert len(view["trucks"]) == 6
    assert len(view["loads"]) == 6
    assert len(view["hubs"]) >= 4
    assert view["timeline"]
    assert view["propagation"]


def test_breakdown_eventually_marks_t03_out():
    view = operations_view("truck_breakdown", severity=0.9, time_hours=2.5)
    t3 = next(t for t in view["trucks"] if t["id"] == "T03")
    assert t3["active"] is False
    assert t3["status"] == "OUT"
