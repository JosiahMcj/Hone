"""Canary tests: every gate must FAIL on synthetic fraud (tautological-test
law) and pass on clean input. These run before every promotion."""
import pytest

from hone.gates import (CONFIG_WHITELIST, GateError, check_frozen_hashes,
                        check_lossless_parity, check_recount,
                        check_wall_sanity, finish_reason_shift,
                        validate_config)


def row(sha="aa", ptok=100, ptps=200.0, gtok=50, gtps=20.0, fr="stop"):
    return {"text_sha256": sha, "prompt_tokens": ptok, "prompt_tps": ptps,
            "generation_tokens": gtok, "generation_tps": gtps,
            "finish_reason": fr}


# --- config whitelist ---

def test_canary_unknown_knob_hard_errors():
    with pytest.raises(GateError, match="whitelist"):
        validate_config({"model": "m", "patch_the_timer": True})


def test_canary_missing_model_hard_errors():
    with pytest.raises(GateError, match="required"):
        validate_config({"num_draft_tokens": 3})


def test_clean_config_passes():
    assert validate_config({"model": "m", "draft_model": "d",
                            "num_draft_tokens": 4, "kv_bits": 8})


def test_whitelist_never_contains_code_knobs():
    assert not any(k in CONFIG_WHITELIST
                   for k in ("runner", "script", "code", "cmd"))


# --- forged token counts ---

def test_canary_forged_count_fails_recount():
    with pytest.raises(GateError, match="recount"):
        check_recount(claimed_tokens=500, recounted_tokens=100, prompt_id="x")


def test_recount_tolerates_detokenization_jitter():
    assert check_recount(claimed_tokens=100, recounted_tokens=99)


# --- forged tps ---

def test_canary_impossible_tps_fails_wall_sanity():
    # claims imply 100/1000 + 50/1000 = 0.15s of compute -- but claims of
    # 1000 tps for both phases against a 10x-too-short wall must die
    fraud = [row(ptps=1000.0, gtps=1000.0, ptok=1000, gtok=500)]
    with pytest.raises(GateError, match="not physically possible"):
        check_wall_sanity(wall_s=0.1, rows=fraud)


def test_honest_wall_passes():
    honest = [row()]  # 100/200 + 50/20 = 3.0s implied
    assert check_wall_sanity(wall_s=3.2, rows=honest)


# --- lossless parity ---

def test_canary_corrupted_output_fails_parity():
    base = {("p1", 1): row(sha="aaa"), ("p2", 1): row(sha="bbb")}
    cand = {("p1", 1): row(sha="aaa"), ("p2", 1): row(sha="CORRUPT")}
    with pytest.raises(GateError, match="parity"):
        check_lossless_parity(base, cand)


def test_identical_outputs_pass_parity():
    base = {("p1", 1): row(sha="aaa")}
    cand = {("p1", 1): row(sha="aaa")}
    assert check_lossless_parity(base, cand)


# --- frozen hashes ---

def test_canary_edited_runner_fails_hash_check():
    with pytest.raises(GateError, match="mismatch"):
        check_frozen_hashes({"runner_sha256": "TAMPERED"},
                            {"runner_sha256": "committed"})


def test_matching_hashes_pass():
    assert check_frozen_hashes({"runner_sha256": "x"}, {"runner_sha256": "x"})


# --- EOS accounting ---

def test_finish_reason_shift_reports_distributions():
    base = {("p1", 1): row(fr="stop"), ("p2", 1): row(fr="stop")}
    cand = {("p1", 1): row(fr="length"), ("p2", 1): row(fr="stop")}
    shift = finish_reason_shift(base, cand)
    assert shift["base"] == {"stop": 2}
    assert shift["cand"] == {"length": 1, "stop": 1}
