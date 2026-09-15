"""Verdict glue: a clear winner passes G1-G4, a fraud/regression fails the
right gate. Gate math itself is tested in test_stats/test_gates."""
import importlib.util
import json
import os

SPEC = importlib.util.spec_from_file_location(
    "verdict", os.path.join(os.path.dirname(__file__), os.pardir,
                            "scripts", "verdict.py"))
verdict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verdict)

MODEL = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"


def make_results(tmp_path, cand_tps=16.0, cand_sha="X", cand_ptps=200.0,
                 reps=6, prompts=3):
    rows = {"base": {}, "cand": {}}
    for r in range(1, reps + 1):
        for p in range(prompts):
            k = f"p{p}|rep{r}"
            rows["base"][k] = {"generation_tps": 12.6, "prompt_tps": 200.0,
                               "text_sha256": "X", "finish_reason": "stop"}
            rows["cand"][k] = {"generation_tps": cand_tps,
                               "prompt_tps": cand_ptps,
                               "text_sha256": cand_sha,
                               "finish_reason": "stop"}
    res = {"header": {"base_config": {"model": MODEL}},
           "rows": rows,
           "walls": {"base": [400.0] * reps,
                     "cand": [400.0 * 12.6 / cand_tps] * reps}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(res))
    return str(path)


def margins_path():
    return os.path.join(os.path.dirname(__file__), os.pardir,
                        "configs", "margins.json")


def test_clear_winner_promotable_pending_g5(tmp_path):
    v = verdict.judge(make_results(tmp_path, cand_tps=16.0), margins_path())
    assert v["g1_pass"] and v["g2_pass"] and v["g3_pass"] and v["g4_pass"]
    assert v["promotable"]
    assert "PENDING" in v["g5"]


def test_screen_mode_never_promotable(tmp_path):
    v = verdict.judge(make_results(tmp_path, cand_tps=16.0), margins_path(),
                      screen=True)
    assert v["g1_pass"] and not v["promotable"]


def test_small_win_fails_g1_margin(tmp_path):
    # +5% is real but below the registered 1.10x margin
    v = verdict.judge(make_results(tmp_path, cand_tps=13.2), margins_path())
    assert not v["g1_pass"] and not v["promotable"]


def test_canary_corrupted_output_fails_g3(tmp_path):
    v = verdict.judge(make_results(tmp_path, cand_tps=16.0, cand_sha="BAD"),
                      margins_path())
    assert not v["g3_pass"] and not v["promotable"]


def test_canary_prefill_regression_fails_g2(tmp_path):
    v = verdict.judge(make_results(tmp_path, cand_tps=16.0, cand_ptps=150.0),
                      margins_path())
    assert not v["g2_pass"] and not v["promotable"]
