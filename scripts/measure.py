"""Measurement orchestrator: interleaved paired arms under the PROTOCOL laws.

Flow per scored rep r (after 1 discarded warmup rep per arm):
  order = (base, cand) if r even else (cand, base)   # ABBA drift cancel
  for each arm: wait_for_quiet_box -> gpulock -> therm snapshot ->
    spawn runner subprocess (wall-clocked) -> therm snapshot ->
    void on therm change (<=2 retries) -> recount + wall sanity gates

Writes runs/<label>/results.json with header (all frozen hashes), rows
keyed (prompt_id, rep), void log, defer counts.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hone import boxsafety  # noqa: E402
from hone.gates import GateError, check_recount, check_wall_sanity  # noqa: E402
from hone.suite import load_suite, sha256_file  # noqa: E402

HONE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PYTHON = os.path.join(HONE_ROOT, ".venv", "bin", "python")
RUNNER = os.path.join(HONE_ROOT, "scripts", "runner.py")
HASH_CACHE = os.path.join(HONE_ROOT, "runs", "model_hashes.json")


def box_id():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True).stdout.strip()
    host = subprocess.run(["hostname", "-s"],
                          capture_output=True, text=True).stdout.strip()
    return f"{host}/{chip}/{int(mem) >> 30}GB"


def model_snapshot_dir(repo_id):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, local_files_only=True)


def model_hash(repo_id):
    """Full sha256 over weight+config files, cached by (size, mtime)."""
    snap = model_snapshot_dir(repo_id)
    files = sorted(
        os.path.join(dp, f) for dp, _, fs in os.walk(snap) for f in fs
        if f.endswith((".safetensors", ".json")) and not f.startswith("."))
    sig = [(os.path.relpath(p, snap), os.path.getsize(p),
            int(os.path.getmtime(p))) for p in files]
    cache = {}
    if os.path.exists(HASH_CACHE):
        with open(HASH_CACHE) as f:
            cache = json.load(f)
    key = repo_id
    if key in cache and cache[key]["sig"] == [list(s) for s in sig]:
        return cache[key]["sha256"]
    h = hashlib.sha256()
    for p, _, _ in ((p, None, None) for p in files):
        h.update(os.path.relpath(p, snap).encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                h.update(b)
    digest = h.hexdigest()
    cache[key] = {"sig": [list(s) for s in sig], "sha256": digest}
    os.makedirs(os.path.dirname(HASH_CACHE), exist_ok=True)
    with open(HASH_CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    return digest


def run_segment(cfg_path, suite_path, out_path, log, runner=RUNNER):
    """One timed segment: guards -> lock -> therm -> runner -> gates.
    Returns (rows, wall_s). Raises GateError on an unrecoverable void."""
    last_void = "unknown"
    for attempt in range(3):
        defers = boxsafety.wait_for_quiet_box(log=log)
        if defers:
            log(f"proceeded after {defers} defers")
        # Loud lock wait: r4/r5 sat silently blocked on the gpulock for
        # 30-47 min and looked dead (probe-holding-lock lesson: log waits).
        t_wait = time.monotonic()
        while True:
            try:
                with boxsafety.gpulock(timeout_s=300):
                    waited = time.monotonic() - t_wait
                    if waited > 5:
                        log(f"gpulock acquired after {waited:.0f}s wait")
                    therm_a = boxsafety.thermal_snapshot()
                    t0 = time.monotonic()
                    env = dict(os.environ, HF_HUB_OFFLINE="1")
                    proc = subprocess.run(
                        [PYTHON, runner, "--config", cfg_path,
                         "--suite", suite_path, "--out", out_path],
                        capture_output=True, text=True, env=env,
                        cwd=HONE_ROOT)
                    wall = time.monotonic() - t0
                    therm_b = boxsafety.thermal_snapshot()
                break
            except TimeoutError:
                log(f"waiting on gpulock "
                    f"({(time.monotonic() - t_wait) / 60:.0f} min so far -- "
                    f"another machine's run holds it)")
        if proc.returncode != 0:
            raise GateError(f"runner failed rc={proc.returncode}: "
                            f"{proc.stderr[-2000:]}")
        if therm_a != therm_b:
            last_void = f"thermal {therm_a} -> {therm_b}"
            log(f"VOID (attempt {attempt}): thermal state changed "
                f"{therm_a} -> {therm_b}; re-running segment")
            continue
        end_reasons = boxsafety.box_quiet_once()
        if end_reasons:
            last_void = f"contention ({'; '.join(end_reasons)})"
            # Amendment A1: contention that ARRIVES mid-segment (start
            # guard passed, box busy by segment end) voids the segment.
            # Cause-based only -- voiding on slow tps is forbidden.
            log(f"VOID (attempt {attempt}): box not quiet at segment end "
                f"({'; '.join(end_reasons)})")
            continue
        rows = []
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["type"] == "row":
                    rows.append(rec)
        return rows, wall
    raise GateError(f"segment voided 3x -- last cause: {last_void} -- deferring")


def recount_tokens(tokenizer, rows):
    for r in rows:
        recount = len(tokenizer.encode(r["text"], add_special_tokens=False))
        check_recount(r["generation_tokens"], recount, r["prompt_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", required=True)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--runner", default=RUNNER,
                    help="runner script; default = single-turn runner.py")
    ap.add_argument("--chat", action="store_true",
                    help="T3-A: multi-turn chat suite + chat runner/loader")
    args = ap.parse_args()
    if args.suite is None:
        args.suite = os.path.join(
            HONE_ROOT, "suites", "t3a", "chat.json") if args.chat else \
            os.path.join(HONE_ROOT, "suites", "t1", "prompts.json")
    if args.chat and args.runner == RUNNER:
        args.runner = os.path.join(HONE_ROOT, "scripts", "runner_chat.py")

    out_dir = os.path.join(HONE_ROOT, "runs", args.label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "measure.log")
    logf = open(log_path, "a")

    def log(msg):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logf.write(f"{stamp} {msg}\n")
        logf.flush()
        print(msg, flush=True)

    if args.chat:
        from hone.chat_suite import load_chat_suite
        _, suite_sha = load_chat_suite(args.suite)
    else:
        _, suite_sha = load_suite(args.suite)
    with open(args.base) as f:
        base_cfg = json.load(f)
    with open(args.cand) as f:
        cand_cfg = json.load(f)
    if base_cfg["model"] != cand_cfg["model"]:
        raise SystemExit("arms must share the target model (per-target law)")

    log(f"hashing frozen artifacts for {args.label}")
    header = {
        "box_id": box_id(),
        "suite_sha256": suite_sha,
        "runner_sha256": sha256_file(args.runner),
        "env_lock_sha256": sha256_file(os.path.join(HONE_ROOT, "env.lock")),
        "model_sha256": model_hash(base_cfg["model"]),
        "base_config": base_cfg, "cand_config": cand_cfg,
        "reps": args.reps, "warmup": args.warmup,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if cand_cfg.get("draft_model"):
        header["draft_model_sha256"] = model_hash(cand_cfg["draft_model"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot_dir(base_cfg["model"]))

    arms = {"base": args.base, "cand": args.cand}
    rows = {"base": {}, "cand": {}}
    walls = {"base": [], "cand": []}

    total_reps = args.warmup + args.reps
    for rep in range(total_reps):
        order = ["base", "cand"] if rep % 2 == 0 else ["cand", "base"]
        scored = rep >= args.warmup
        for arm in order:
            tag = f"{arm}-rep{rep}{'' if scored else '-warmup'}"
            out_path = os.path.join(out_dir, f"{tag}.jsonl")
            log(f"segment {tag} starting")
            seg_rows, wall = run_segment(arms[arm], args.suite,
                                         out_path, log, runner=args.runner)
            recount_tokens(tokenizer, seg_rows)
            check_wall_sanity(wall, seg_rows)
            log(f"segment {tag} ok: wall {wall:.1f}s, "
                f"median decode tps "
                f"{sorted(r['generation_tps'] for r in seg_rows)[len(seg_rows)//2]:.2f}, "
                f"end pressure={boxsafety.pressure_level()} "
                f"load={boxsafety.loadavg_1min():.1f}")
            if scored:
                walls[arm].append(wall)
                for r in seg_rows:
                    rows[arm][f"{r['prompt_id']}|rep{rep}"] = r

    header["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"header": header, "rows": rows, "walls": walls}, f,
                  indent=1)
    log(f"results written to {out_dir}/results.json")


if __name__ == "__main__":
    main()
