"""Load-guard behavior with injected probes -- production never exits on a
busy box, alarms on consecutive defers, and double-probes quiet readings."""
import pytest

from hone import boxsafety


def test_waits_through_busy_then_proceeds():
    seq = iter([4, 4, 1, 1])  # two busy probes, then quiet double-probe
    sleeps = []
    defers = boxsafety.wait_for_quiet_box(
        log=lambda m: None,
        pressure_fn=lambda: next(seq),
        load_fn=lambda: 0.5,
        sleep_fn=sleeps.append)
    assert defers == 2
    assert boxsafety.DEFER_SLEEP_S in sleeps


def test_double_probe_catches_flapping_box():
    # quiet on first probe, busy on the confirm probe -> must defer
    seq = iter([1, 4, 1, 1])
    defers = boxsafety.wait_for_quiet_box(
        log=lambda m: None,
        pressure_fn=lambda: next(seq),
        load_fn=lambda: 0.5,
        sleep_fn=lambda s: None)
    assert defers == 1


def test_alarm_fires_every_10_defers_and_never_exits():
    msgs = []
    with pytest.raises(TimeoutError):  # max_defers is a TEST-ONLY escape
        boxsafety.wait_for_quiet_box(
            log=msgs.append,
            pressure_fn=lambda: 4,
            load_fn=lambda: 0.5,
            sleep_fn=lambda s: None,
            max_defers=25)
    alarms = [m for m in msgs if m.startswith("ALARM")]
    assert len(alarms) == 2  # at defer 10 and 20
    assert "25" not in "".join(alarms[:1])


def test_loadavg_alone_defers():
    reasons = boxsafety.box_quiet_once(pressure_fn=lambda: 1,
                                       load_fn=lambda: 99.0)
    assert any("loadavg" in r for r in reasons)


def test_quiet_box_no_reasons():
    assert boxsafety.box_quiet_once(pressure_fn=lambda: 1,
                                    load_fn=lambda: 0.5) == []


def test_chronic_pressure_2_does_not_defer():
    # Amendment A2: level 2 is this box's chronic ambient state (r3
    # deferred 57 consecutive times on an idle box, loadavg 1.6); only
    # CRITICAL (4) defers. Contamination tracks loadavg, not warm caches.
    assert boxsafety.box_quiet_once(pressure_fn=lambda: 2,
                                    load_fn=lambda: 1.5) == []


def test_critical_pressure_still_defers():
    reasons = boxsafety.box_quiet_once(pressure_fn=lambda: 4,
                                       load_fn=lambda: 1.5)
    assert any("pressure" in r for r in reasons)


def test_thermal_snapshot_parses_pmset_output(monkeypatch):
    class FakeProc:
        stdout = ("Note: No thermal warning level has been recorded\n"
                  "CPU_Scheduler_Limit \t= 100\n"
                  "CPU_Available_CPUs \t= 10\n")
    monkeypatch.setattr(boxsafety.subprocess, "run",
                        lambda *a, **k: FakeProc())
    snap = boxsafety.thermal_snapshot()
    assert snap == {"CPU_Scheduler_Limit": 100, "CPU_Available_CPUs": 10}
