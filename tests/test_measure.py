"""run_segment must VOID a segment when the box is no longer quiet at
segment END (contention that arrives mid-segment, after the start-of-
segment guard passed -- the r2 calibration hole, Amendment A1)."""
import importlib.util
import json
import os

import pytest

SPEC = importlib.util.spec_from_file_location(
    "measure", os.path.join(os.path.dirname(__file__), os.pardir,
                            "scripts", "measure.py"))
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


class NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_contention_at_segment_end_voids_and_retries(monkeypatch, tmp_path):
    calls = {"runner": 0}
    # first post-segment probe: busy (must void); second: quiet (must pass)
    end_states = iter([["loadavg 12.0 > 6.0"], []])
    logs = []

    monkeypatch.setattr(measure.boxsafety, "wait_for_quiet_box",
                        lambda log: 0)
    monkeypatch.setattr(measure.boxsafety, "gpulock",
                        lambda **kw: NullLock())
    monkeypatch.setattr(measure.boxsafety, "thermal_snapshot", lambda: {})
    monkeypatch.setattr(measure.boxsafety, "box_quiet_once",
                        lambda: next(end_states))

    out_path = str(tmp_path / "seg.jsonl")

    def fake_runner(cmd, **kw):
        calls["runner"] += 1
        with open(out_path, "w") as f:
            f.write(json.dumps({"type": "header"}) + "\n")
            f.write(json.dumps({"type": "row", "prompt_id": "p1",
                                "generation_tps": 40.0}) + "\n")
        class P:
            returncode = 0
            stderr = ""
        return P()

    monkeypatch.setattr(measure.subprocess, "run", fake_runner)

    rows, wall = measure.run_segment("cfg.json", "suite.json", out_path,
                                     logs.append)
    assert calls["runner"] == 2, "contaminated segment must be re-run"
    assert any("not quiet at segment end" in m for m in logs)
    assert rows[0]["prompt_id"] == "p1"
