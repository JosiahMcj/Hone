# Hone -- design

One machine, one objective at a time. Maximize LLM inference throughput
on my own hardware at strict, verifiable quality parity. Targets are
pluggable modules. The first target (T1) is MLX decode speed on an M4
Mac mini (24GB). "Go after something else next" means writing a new
target module, not a new machine.

## Null hypothesis

Above stock `mlx-lm` defaults, tuned configuration buys less than 1.2x
median decode speedup at quality parity on this box. A powered negative
-- an honest headroom map showing the stock path is already near the
wall -- is an acceptable, publishable outcome.

## Non-goals (v1)

- No custom Metal kernels. I tune the space ABOVE the kernels.
- No model quality changes, no training, no fine-tuning.

## Target module contract

A target = { benchmark suite, pinned baseline, quality gate, search
space, noise model }. Adding a target means shipping all five, hashed,
before any tuning run against it.

## T1: MLX decode speed

- **Models:** Qwen2.5-Coder-14B-Instruct-4bit (primary -- the model I
  actually run daily) and Qwen3-4B-Instruct-2507-4bit (small class),
  content-hashed. Tuning transfers poorly across model classes, so both
  get measured.
- **Prompt suite:** fixed and hashed. Short chat, code generation, and
  long-context (~6k-token) document prompts, fixed max_tokens, greedy.
- **Baseline:** stock `mlx_lm` generation, pinned package version,
  default parameters. Version bumps void all prior comparisons.
- **Metrics:** prefill tokens/sec and decode tokens/sec reported
  SEPARATELY, since they trade off differently. Suite wall-clock is the
  tiebreaker.
- **Search space (v1 knobs):** speculative decoding (draft model
  choice, number of draft tokens), quantized KV cache, prompt-cache
  reuse, chunked prefill size, wired-memory limit, sampler path,
  compilation coverage.

## Quality parity gates (two technique classes)

- **Lossless class** (greedy + speculative decoding). Output must be
  byte-identical to baseline. Hard gate. Calibration FIRST measures the
  baseline's own run-to-run byte-stability on Metal. The gate is
  calibrated to measured stability, never assumed.
- **Lossy class** (quantized KV, etc.). Score parity on a frozen task
  benchmark within a margin registered from A/A' noise before first
  use. Never eyeballed.

## Measurement laws (pre-registered, full text in PROTOCOL.md)

- Paired interleaved A/B, discarded warmup, medians over repetitions,
  PAIRED differences only, margins set at 2-3x measured A/A' noise.
- Promotion decisions use an exact sign test over discordant pairs.
  Underpowered comparisons are excluded, never counted as negatives.
- Box discipline. Timed segments hold an exclusive GPU lock, defer on
  real load (never exit), and void on thermal-state changes and on
  contention that arrives mid-segment. Cause-based voiding only --
  voiding a segment because its numbers look slow biases the noise
  floor and is forbidden.
- Timer integrity. Wall-clock is measured by the harness around a
  subprocess. Token counts are recomputed by the tokenizer from captured
  output, never taken from the runner's self-report.

## Promotion gate (conjunctive, mechanical, no human judgment inside)

- G1 decode speedup >= registered margin (paired).
- G2 prefill / suite wall-clock not worse beyond registered allowance.
- G3 quality parity gate for the technique's class.
- G4 integrity -- all hashes match, lock held, guards green, recounts and
  wall-clock sanity pass, contamination check clean.
- G5 stability -- replicates in a separate session under a different
  contention regime before promotion.

## Wall verdict procedure

A knob or target is declared a WALL only after a full-effort headroom
probe -- the published best-known technique at its best-known
configuration, run under the measurement laws. "I couldn't tune it"
is not "impossible".

## Anti-gaming guards

- A candidate is a CONFIG JSON validated against a whitelist of knobs.
  Unknown keys are a hard error. Candidates never supply code.
- Runner and harness file hashes are recorded in every run header. A
  mismatch voids the run.
- Canary gates. Before the first real run, a deliberately broken config
  must FAIL each gate once, and the parity gate must catch a
  deliberately corrupted output. Every new assertion fails once on
  purpose before it is trusted.
- max_tokens-reached vs natural stop is logged per generation. A
  "faster" run that silently generated fewer tokens is voided by the
  token recount.

## Phases

- **P0** -- harness + baseline + noise calibration. Exit when margins
  are registered by amendment and canary gates are demonstrated failing.
- **P1** -- manual technique sweeps producing an honest headroom map.
  Exit rule, registered in advance: if the sweep saturates the
  headroom, the automated search loop of P2 is theater and I say so.
- **P2** -- a self-tuning search loop, built ONLY if P1 shows a
  combinatorial space worth searching. (It did not -- see RESULTS.md.)

## Beyond T1: a second box and three more target modules

The machine is target- and box-agnostic by construction, so I applied it
further, each time honoring the same laws.

- **Second box (T2).** I re-ran the whole calibration on a second
  machine (M3 Ultra, 96GB) from scratch, under the per-box law. No
  margin, no baseline, and no quality-stability result transfers across
  hardware. New target models get their own A/A' before anything is
  judged.
- **New target modules (T3).** A target module is defined by its metric
  and gate, not by the model. Beyond greedy short-context DECODE I added
  three more -- multi-turn PREFILL (time-to-first-token, for agent loops
  that re-send a large prefix), SAMPLED decode (temperature > 0, where
  the byte-compare gate is replaced by the distribution-preservation
  guarantee of speculative sampling), and LONG-CONTEXT decode (a suite
  of ~9k-token prompts). Each ships its own frozen suite, margin, and
  gate, registered before use.

The findings from all of this are in `RESULTS.md`. The point here is
that "go after something else" always means a new module under the same
measurement discipline, never a loosening of it.
