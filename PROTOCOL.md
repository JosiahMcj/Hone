# Hone PROTOCOL (pre-registered 2026-08-03, before any experiment)

This file registers the measurement laws BEFORE the first timed run.
Amendments are append-only, dated, and written before the results they
govern. Design context: `DESIGN.md`.

## Decisions locked 2026-08-03

- **T1 models:** `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`
  (primary -- my daily-driver class) +
  `mlx-community/Qwen3-4B-Instruct-2507-4bit` (small class).
  Speculative-decoding drafts: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
  and `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` (same Qwen2.5
  vocabulary as the 14B; the Qwen3-4B had no compatible cached draft at
  registration -- see A7).
- **Primary metric:** decode tokens/sec. Prefill tokens/sec and suite
  wall-clock are guard metrics, reported separately, never averaged in.
- **Success framing:** the honest headroom map is the deliverable. The
  promotion margin comes from measured A/A' noise; 1.5x is a stretch
  FLAG, not a gate. A powered negative is a valid outcome.

## Per-box law

Noise, margins, and guard thresholds are PER-BOX. Every run header
records box_id = (hostname, chip, RAM). This box: Mac mini M4, 24GB.
These margins transfer nowhere else.

## Frozen artifacts (hashed into every run header; any mismatch voids)

suite JSON sha256; model weight-file sha256 set (full-file hash, cached
by path+size+mtime); `env.lock` sha256 (mlx 0.32.0, mlx-lm 0.31.3,
transformers 5.14.1, Python 3.14.3); `scripts/runner.py` sha256;
mlx/mlx-lm versions echoed by the runner itself.

## Measurement laws

- **Timed segment** = one runner subprocess pass over the full suite
  for one arm. An exclusive fcntl lock on a shared GPU-lock file is
  held per segment and released between segments so other jobs on the
  box are never starved.
- **Load guard (DELAYS, never exits):** proceed only when the box is
  quiet on a DOUBLE probe (5s apart). On defer: sleep 30s, log the
  reason, ALARM every 10 consecutive defers. Other training jobs are
  detected by command-line pattern, never by hardcoded instance names,
  and are never signaled.
- **Thermal void:** thermal state sampled before/after every timed
  segment; any change voids the segment (max 2 retries). Voids are
  logged, never silently averaged.
- **Interleaving:** repetition r runs arms in order (A,B) for even r,
  (B,A) for odd r -- ABBA drift cancellation. The pair is
  (prompt, repetition).
- **Warmup:** repetition 0 of every arm is discarded. Registered here,
  not decided post hoc.
- **Repetitions:** R = 6 scored (+1 warmup) for calibration and
  promotions.
- **Token recount (forged-count defense):** the harness re-tokenizes
  the captured output; |runner claim - recount| <= max(2 tokens, 2%) or
  the segment is VOIDED. Throughput claims are cross-checked against
  subprocess wall-clock: wall >= 0.95 x implied compute time or VOIDED.
- **Stop accounting:** finish_reason recorded per generation; a config
  whose "speedup" coincides with a shift in the stop distribution is
  flagged and not creditable until explained.
- **Stats:** per-(prompt, rep) paired relative differences on decode
  tokens/sec; verdicts use the exact sign test over DISCORDANT pairs;
  MIN_DISCORDANT = 8; anything under that is UNDERPOWERED -- excluded,
  never counted as negative.
- **Margins:** set ONLY by amendment after A/A' calibration, at 2-3x
  measured paired noise, floor 1.10x. Until that amendment exists, no
  comparison may be called a win.

## Quality parity (gate G3; two classes)

- **Lossless** (greedy, speculative decoding): byte-identical output
  per (prompt, rep) vs baseline, with the baseline's own byte-stability
  measured first.
- **Lossy** (quantized KV, etc.): frozen-benchmark score parity within
  a margin registered before first use.

## Promotion gate

G1 decode >= margin (paired) | G2 prefill/wall within allowance |
G3 parity for the class | G4 integrity (hashes, lock, guards, recount,
wall sanity, contamination) | G5 replicates in a separate
session/contention regime.

## Anti-gaming guards

- Candidates are config JSON against a whitelist (draft_model,
  num_draft_tokens, kv_bits, kv_group_size, quantized_kv_start,
  max_kv_size, prefill_step_size, wired_limit_gb). Unknown keys are a
  hard error. Candidates never supply code.
- Runner/harness hashes recorded per run; mismatch voids.
- Canary tests (run before every promotion): corrupted output must fail
  parity; forged token counts must fail the recount; forged throughput
  must fail wall sanity; an unknown config knob must hard-error; an
  edited suite must change the suite hash. Each canary asserts the gate
  FAILS -- every new assertion fails once on purpose.

## Amendments (append-only, dated)

### A1 (2026-08-04): first calibration INVALID; post-segment guard law; byte-stability registered

- **The first A/A' run was INVALID for margins.** The pre-registered
  sanity rule fired: worst identical-vs-identical pair showed a fake
  "1.475x speedup" against a 0.163 limit.
- **Diagnosis (cause, not outcome):** all 10 extreme pairs sat in two
  repetitions whose segments collapsed to 17-22 tokens/sec (one prompt
  at 2.0) while all other segments sat at 40-42. Contention from
  processes that do not take the GPU lock arrived MID-segment, after
  the start-of-segment guard had passed.
- **New law:** the box-quiet probe re-runs immediately after each
  segment; ANY reason voids the segment. Voiding is CAUSE-BASED only:
  voiding a segment because its throughput looks slow is outcome
  selection, biases the noise floor down, and is forbidden. (Canary
  test failed once against the unpatched orchestrator.)
- **Finding that SURVIVES the invalid run (load-independent):**
  baseline byte-stability is PERFECT (12/12 prompts, one distinct
  output hash each across 14 segments). The lossless gate is hereby a
  HARD byte-compare.
- **Recorded evidence:** the A/A' sign test came out "significant"
  (p=0.024) on identical configs under contamination -- a p-value alone
  can false-positive under drift; the conjunctive margin requirement
  stays mandatory.

### A2 (2026-08-04): pressure-guard threshold corrected -- level 2 is ambient, not risk

- **Evidence:** a rerun deferred 57 consecutive times on an otherwise
  idle box (loadavg 1.6) because the guard required macOS memory
  pressure level 1. Meanwhile every clean 40-42 tokens/sec segment in
  the invalid run had happened DURING ambient level 2, and the actual
  contamination coincided with loadavg spikes of 9-15. The guard was
  measuring cache warmth, not risk.
- **Law change:** defer/void on loadavg > 6.0 (double probe) or
  pressure level CRITICAL (4). Level 2 alone no longer defers. Canary
  failed once against the unpatched guard; critical-still-defers has
  its own test.
- **Falsifiable escape hatch:** if a future calibration fires the
  sanity rule with contamination correlated to pressure-2 windows
  rather than loadavg, this threshold returns to review.
- **Audit:** every segment log line records end-of-segment pressure and
  loadavg.

### A3 (2026-08-10): sanity rule replaced (mine was mis-scaled); 4B margins REGISTERED

- **My registered sanity rule was wrong, said out loud:** it compared a
  SINGLE-PAIR extreme against the MEDIAN-based margin. The median over
  72 pairs has a standard error near 0.2%; a lone +14% pair cannot move
  it. The rule fired on clean data for a scale-mismatch reason and
  would fire on any realistic tail forever.
- **Replacement rule (BACKTESTED through the code path on data already
  in hand before registration):** contamination iff any segment's
  suite-median decode tokens/sec deviates >15% from the grand median
  (`hone.stats.contamination_check`, canary tests both directions).
  Backtest: the invalid run fires on exactly its three known-bad
  segments (-57%, -46%, -27%); the clean run passes (worst -8.2%).
  Band rationale: observed clean ambient wobble tops out ~8%; the
  observed contamination mode is 40-60%.
- **Registered from the clean run (Qwen3-4B, R=6, n=72):** A/A' paired
  noise median +0.21%, MAD 1.16%, sign test properly null (p=0.289).
  **G1 margin: >= 1.10x median paired decode speedup** (floor binds).
  **G2 allowance: <= 5% prefill/wall regression.** Byte-stability
  perfect again -- lossless G3 stands as byte-compare.

### A4 (2026-08-10): 14B margins REGISTERED; screening law

- **Registered from a clean 14B A/A' (R=6, n=72, contamination
  CLEAN):** paired noise median -0.16%, **MAD 0.76%**; sign test null
  (p=0.076). **G1: >= 1.10x (floor binds). G2: <= 5%.** Baseline decode
  ~12.65 tokens/sec; byte-stability perfect on this model too.
- **Screening law (registered before any screening run):** candidate
  ranking may use paired R=2 screening runs. Screening produces NO wins
  and NO reportable speedups -- it only selects candidates for full
  R=6 promotion runs. Screening runs are labeled `screen-*`.
- **P1 wave 1 candidates (spec decode, 14B target, greedy):** 0.5B
  draft x num_draft_tokens {2,3,4,6,8}; 3B draft x {2,4,6}. Registered
  margins live in `configs/margins.json`.

### A5 (2026-08-11): per-arm contamination; spec-decode divergence policy -- registered BEFORE the R=6 promotion runs

- **Contamination check is now PER-ARM.** The pooled A3 rule flagged
  all three screening winners: a candidate legitimately +41..79% faster
  deviates from the POOLED median by design. Deviations are now judged
  against each arm's own grand median. Canaries failed once against the
  pooled code; backtest on all six datasets: the invalid run still
  fires on exactly its three bad segments, everything clean stays
  clean.
- **Screening wave-1 result (ranking only):** 0.5B draft k=2 > k=3 >
  k=4 (all margin-clearing); the 3B draft pays a prefill tax
  (-15..-30%, G2 fail); k>=6 over-drafts to a loss on both drafts.
- **Spec-decode byte-divergence, measured:** 2/12 prompts diverge
  DETERMINISTICALLY -- identical position every repetition and every k,
  both arms individually byte-stable. Content deltas are trivial (an
  article; an inserted comment). Teacher-forced top-2 gaps at the flip
  positions measure ~0.4-1.5 nats via character-prefix re-tokenization
  (imprecision acknowledged); decisive control positions measure 6-13
  nats. Mechanism: batched-verification fp16 numerics flip genuine
  near-tie argmaxes onto alternate model-sanctioned greedy paths.
- **Instrumentation:** the runner now records exact generated
  `token_ids` per row (runner hash changes here); the divergence probe
  uses exact token prefixes when present.
- **G3-spec policy (mechanical, registered now; no gap threshold -- two
  observations do not calibrate one):** a spec-decode candidate passes
  G3 iff byte-parity holds, OR all of: (i) diverged prompts <= 1/3 of
  the suite; (ii) each arm individually byte-stable across all reps;
  (iii) stop-reason distribution unshifted; (iv) the exact-token
  divergence-probe report is attached to the promotion record. Any
  promotion under this clause is reported as "quality: near-tie-flip
  class, NOT byte-exact" -- never as lossless.

### A6 (2026-08-12): promotion recorded; P1 wave 2 registered

- **spec-05b-k2 PROMOTED** (RESULTS.md): G1-G4 on the primary R=6 run,
  G5 replication +77.1% with an identical deterministic divergence
  signature.
- **Wave 2 candidates (registered before any run):** on top of the
  champion: kv_bits {8,4}; prefill_step_size {512, 4096};
  wired_limit_gb 18; plus kv8 without a draft, to isolate the KV
  effect. Purpose: interaction map -- decides whether an automated
  search loop has a combinatorial case or the wave-1 grid was
  effectively one-dimensional.
- **Lossy-class caveat:** kv-quantized candidates may SCREEN for speed
  but CANNOT promote under A5's near-tie clause -- KV quantization
  perturbs every step, not just coin-flip tokens. Their promotion
  requires the lossy-class frozen-benchmark gate, which does not exist
  yet; building it is a registered precondition.

### A7 (2026-08-13): P1 CLOSED; the P2 search loop DECLINED per the exit law; wave 3

- **Wave-2 verdicts (vs champion):** prefill_step 512/4096 = +-0.3%
  (noise; byte-identical); champion+kv4 = -24.5% decode; kv8-solo =
  -44.3% vs champion (i.e., ~zero decode effect vs plain baseline).
  champion+kv8 completed 10/12 prompts ~10% slower, then FAULTED Metal
  on the second long-context prompt (repro: clean on short context --
  load-dependent fault). champion+wired18 is an invalid config:
  set_wired_limit rejects 18GB > max working set on the 24GB M4.
- **P2 verdict:** wave 1 was a 1-D grid with a sharp optimum at k=2;
  wave 2 shows NOTHING stacks on the champion. There is no
  combinatorial surface on T1 -- the automated search loop is DECLINED
  per the pre-registered exit law.
- **Wave 3 (final T1 sweep):** cross-class check on the small model
  with the smallest in-family compatible draft,
  `mlx-community/Qwen3-0.6B-4bit` (downloaded + hashed on first use).
  Candidates k={2,3}, R=2 screen; full R=6 only if the screen clears
  the margin. Verdict recorded in RESULTS.md.

## Extensions: a second box and the T3 target modules

These amendments extend the method to a second machine and to three new
target modules. Each was registered before the comparison it governs;
the ops details of running on shared hardware are omitted here as they
are not part of the method.

### A-T2a: second box calibrated from scratch (per-box law)

I re-ran P0 on a second box (M3 Ultra, 96GB): fresh A/A' noise, fresh
byte-stability, fresh margins for every target model. Nothing from the
first box transfers. Recorded floors there: 30B-A3B MoE MAD 1.7%,
dense-32B MAD 0.16%; both byte-stable, both binding on the 1.10x / 5%
floors.

### A-T2b: the ratio law, sharpened to step time

The first box showed speculative decoding pays only at small draft/target
size ratios. The second box's MoE result forced the correction: with a
healthy draft acceptance rate the MoE target still lost decode, because
what matters is the draft's per-step wall time relative to the TARGET's
per-step wall time -- governed by ACTIVE parameters and box speed, not
total parameter count. Registered as the operative form of the law.

### A-T3a: prompt-cache reuse -- the prefill metric and its margin

New target module for multi-turn / agent loops. Primary metric is
`prefill_s` (time-to-first-token), recorded directly, not derived from a
throughput number (the two arms process different token counts, so a tps
ratio is not comparable). The promotion margin is RELATIVE and registered
from A/A' as before; the relative form is valid because the baseline
(full re-prefill) is never near zero. Cache-reuse correctness is a hard
byte-compare, and I verified the underlying cache semantics with a
small-model probe before trusting any timing.

### A-T3b: sampled decode -- the quality gate at temperature > 0

For temperature > 0 the byte-compare gate does not apply: speculative
decoding is not sample-path-identical to plain sampling even at a matched
seed (the RNG is consumed differently). The registered gate there is
distribution preservation -- a proven property of speculative sampling --
supported by two checks I ran first: seeded sampling is reproducible on
this stack, and each arm is per-seed deterministic. Speed is still the
paired decode-tps sign test.

### A-T3c: long-context decode -- a new frozen suite, same gates

A frozen suite of ~9k-token-context prompts, same decode metric and
greedy byte-compare gate. Registered to test whether the win survives
when both the target's attention cost and the draft's own context cost
grow; the prefill guard (G2) is what catches the draft's added cost.
