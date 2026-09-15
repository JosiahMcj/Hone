"""Shared-box discipline for timed runs (24GB M4 mini, parallel jobs).

Laws in PROTOCOL.md:
- gpulock: exclusive fcntl.flock on ~/.gpulock, flock(1)-compatible, held
  per timed segment only.
- load guard: memorystatus pressure level + loadavg, DOUBLE probe, DELAYS
  forever (never exits), alarm every 10 consecutive defers.
- swap-free fraction is NOT a jetsam proxy on macOS (I measured a guard
  that deferred for days on a healthy warm box) -- pressure level is
  used instead.
- other training jobs are detected by command-line PATTERN, never by
  hardcoded instance names, and are never signaled. Edit the pattern
  list for the jobs on your own box.
"""
import contextlib
import fcntl
import os
import subprocess
import time

LOCK_PATH = os.path.expanduser("~/.gpulock")
TRAINER_PATTERNS = ["/trainer/", "train.py", ".venv/bin/python"]

# Amendment A2: defer only on CRITICAL pressure (4). Level 2 (warn) is
# this box's chronic ambient state -- a rerun deferred 57 consecutive
# times on an idle box (loadavg 1.6) requiring ==1, while every clean
# 40-42 tps segment ran during ambient level 2. Contamination tracks
# loadavg.
PRESSURE_CRITICAL = 4
LOADAVG_MAX = 6.0
PROBE_GAP_S = 5
DEFER_SLEEP_S = 30
ALARM_EVERY = 10


@contextlib.contextmanager
def gpulock(lock_path=LOCK_PATH, timeout_s=None):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        if timeout_s is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            deadline = time.time() + timeout_s
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        raise TimeoutError(f"gpulock busy > {timeout_s}s")
                    time.sleep(2)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def pressure_level():
    """macOS jetsam pressure: 1 normal, 2 warn, 4 critical."""
    out = subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True, text=True).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 4  # unreadable -> treat as critical, defer


def loadavg_1min():
    return os.getloadavg()[0]


def other_trainers_active():
    out = subprocess.run(["ps", "-axo", "command"],
                         capture_output=True, text=True).stdout
    return any(p in out for p in TRAINER_PATTERNS)


def thermal_snapshot():
    """Parse `pmset -g therm` into {key: int}. Empty dict = no limits
    recorded (the healthy state). Any change across a timed segment voids
    it (PROTOCOL thermal-void law)."""
    out = subprocess.run(["pmset", "-g", "therm"],
                         capture_output=True, text=True).stdout
    snap = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            try:
                snap[key.strip()] = int(val.strip())
            except ValueError:
                continue
    return snap


def box_quiet_once(pressure_fn=pressure_level, load_fn=loadavg_1min):
    reasons = []
    p = pressure_fn()
    if p >= PRESSURE_CRITICAL:
        reasons.append(f"memory pressure level {p}")
    load = load_fn()
    if load > LOADAVG_MAX:
        reasons.append(f"loadavg {load:.1f} > {LOADAVG_MAX}")
    return reasons


def wait_for_quiet_box(log=print, pressure_fn=pressure_level,
                       load_fn=loadavg_1min, sleep_fn=time.sleep,
                       probe_gap_s=PROBE_GAP_S, max_defers=None):
    """DELAYS until the box is quiet on a DOUBLE probe. Never exits on a
    busy box (the sys.exit guard killed a 9h campaign). max_defers exists
    for tests only; production passes None."""
    defers = 0
    while True:
        reasons = box_quiet_once(pressure_fn, load_fn)
        if not reasons:
            sleep_fn(probe_gap_s)
            reasons = box_quiet_once(pressure_fn, load_fn)
            if not reasons:
                return defers
        defers += 1
        if defers % ALARM_EVERY == 0:
            log(f"ALARM: hone deferred {defers} consecutive times "
                f"({'; '.join(reasons)}) -- box has not gone quiet")
        else:
            log(f"defer {defers}: {'; '.join(reasons)}")
        if max_defers is not None and defers >= max_defers:
            raise TimeoutError(f"box not quiet after {defers} defers")
        sleep_fn(DEFER_SLEEP_S)
