# Hone RESULTS

Box: Mac mini M4, 24GB. Stack: mlx 0.32.0 / mlx-lm 0.31.3 / Python
3.14.3, pinned in `env.lock`. All laws, margins, and amendments are in
`PROTOCOL.md`. Nothing below was judged before its margins were
registered. Raw run data for every headline number is in `evidence/`.

## P0 (2026-08-03 .. 08-10): the measurement floor

- A/A' paired noise (R=6, n=72 pairs, contamination-clean):
  Qwen3-4B MAD 1.16%, Qwen2.5-Coder-14B MAD 0.76%. The 1.10x promotion
  floor binds on both models. Guard allowance 5%.
- Greedy decode is byte-stable run-to-run on this stack. 42/42 clean
  segments across three runs, both models, so the lossless quality
  gate is a hard byte-compare.
- Three protocol defects were caught by the machinery itself and fixed
  by amendment. A1 was mid-segment contention. A2 was a memory-pressure
  guard that measured cache warmth, not risk. A3 was a sanity rule I had
  mis-scaled. Each fix shipped with a canary test that failed once
  against the unpatched code, and each was backtested on data already
  in hand.

## P1 wave 1 (08-10 .. 08-12): speculative decoding, 14B target

Target: Qwen2.5-Coder-14B-Instruct-4bit, baseline 11.88 tokens/sec
median decode. Greedy. Frozen 12-prompt suite (chat / code /
long-context).

### Headroom map

| candidate | screen R=2 | full R=6 | prefill | verdict |
|---|---|---|---|---|
| 0.5B draft, k=2 | +79.3% | **+75.0% (p=4.2e-22)** | +0.4% | **PROMOTED** |
| 0.5B draft, k=3 | +64.1% | +62.7% (p=4.2e-22) | -3.1% | passes gates, dominated by k=2 |
| 0.5B draft, k=4 | +41.1% | -- | -3.9% | screened out |
| 0.5B draft, k=6 | -2.4% | -- | -13.2% | over-draft wall |
| 0.5B draft, k=8 | -13.2% | -- | -18.9% | over-draft wall |
| 3B draft, k=2 | +49.5% | -- | -15.0% | draft prefill tax fails guard |
| 3B draft, k=4 | +28.9% | -- | -22.2% | draft prefill tax fails guard |
| 3B draft, k=6 | -5.6% | -- | -29.8% | prefill tax + over-draft |

Negative findings are results. The 3B draft's own prompt-processing
cost fails the prefill guard everywhere, and drafting more than ~3
tokens ahead loses outright on this model and box.

### The promoted configuration: spec-05b-k2

Qwen2.5-0.5B-Instruct-4bit draft, num_draft_tokens=2, greedy
(`configs/spec-05b-k2.json`, usable via `bin/hone-fast`).

- **Decode: 11.88 -> 21.04 tokens/sec median (+75.0%)**, paired over 72
  (prompt, rep) pairs, sign test p = 4.2e-22, all 72 pairs positive on
  the full run. Suite wall-clock -29.4%. Prefill +0.4%.
- **Quality: near-tie-flip class, NOT byte-exact** (A5 policy). 10/12
  prompts byte-identical across all reps. 2/12 diverge deterministically
  at one token each. The exact-token-prefix probe measures the top-2
  gap at those positions at **0.016 nats** (decisive control positions:
  4.3-13.5 nats). The mechanism is batched-verification fp16 numerics
  flipping genuine coin-flip argmaxes onto alternate model-sanctioned
  greedy paths. Both arms individually byte-stable. Stop-reason
  distribution unshifted.
- **Contention audit.** A later review found that another job had run
  unserialized during the primary run's window (holding a lock file
  descriptor open is not holding the lock). A per-segment audit found all
  12 scored segments within +-7% of their arm median (invalidation band
  15%, and clean A/A' wobble reaches 8%). The one notable dip is
  symmetric across arms, which pairing absorbs. The post-segment guard
  VOIDed and re-ran 2 segment attempts during exactly that window. The
  guards judge box state, not the lock -- by design.
- **G5 replication** in a separate session ~14h later, a different
  contention regime, gave **+77.1%** (p=4.2e-22), prefill +0.2%, and an
  IDENTICAL divergence signature -- same 2 prompts, same token indices,
  same 0.016-nat gaps, deterministic across days. The evidence chain
  spans three time regimes across three days (screen, night primary,
  midday replication).

## Wave 2 (08-12): interactions vs the champion -- ALL NULL OR WORSE

| candidate (on champion) | decode vs champion | note |
|---|---|---|
| prefill_step 512 / 4096 | +0.1% / +0.3% | noise, byte-identical |
| + kv4 | -24.5% | prefill -12.3% too |
| kv8 solo (no draft) | -44.3% | = plain baseline, KV quant is a memory knob, not speed |
| + kv8 | ~-10% where it ran | FAULTED Metal on the 2nd long-context prompt (repro: short context clean) |
| + wired_limit 18GB | invalid | set_wired_limit > max working set on a 24GB M4 |

## Wave 3 (08-13): cross-class check on the 4B -- NO PROMOTABLE CONFIG

Qwen3-4B + Qwen3-0.6B draft, the smallest in-family. k=2 = +18.4% decode
(p=0.0015, clears the speed margin) but **prefill -16.9% (guard
FAIL)**. k=3 = +10.5% (margin fail), prefill -17.8%. The draft-prefill
tax that disqualified the 3B draft on the 14B is fatal on the small
class. Registered negative -- **the spec-decode win is specific to
large-target / tiny-draft ratios** (0.5B/14B = 3.5% of target size
works, while 0.6B/4B = 15% and 3B/14B = 21% both fail the guard).

## T1 COMPLETE, and the P2 search loop DECLINED (2026-08-13)

One promotion (+75-77%, replicated), one registered small-class
negative, five mapped walls. Wave 1 was a one-dimensional grid with a
sharp optimum. Wave 2 showed nothing stacks on it. I registered the
exit rule before the sweep. If the grid saturates the headroom, an
automated search loop is theater. It did, so I declined to build it.

## T2 (second box, M3 Ultra 96GB): larger targets

I moved the machine to a second box (M3 Ultra, 96GB) and re-calibrated
from scratch -- per-box law, so nothing here inherits the mini's
margins. A/A' noise (decode tps): 30B-A3B MAD 1.7%, dense-32B MAD 0.16%,
both byte-stable, both binding on the 1.10x speed floor and 5% prefill
allowance.

### A mixture-of-experts negative that sharpened the law

Target Qwen3-Coder-30B-A3B (a ~3B-active MoE), baseline ~105 tps decode.
I screened 0.6B and 1.7B drafts at k={2,3}. Every one LOST decode.

| candidate | decode | prefill | verdict |
|---|---|---|---|
| 0.6B k=2 | -13.6% | -20.6% | negative |
| 0.6B k=3 | -22.4% | -24.5% | negative |
| 1.7B k=2 | -16.9% | -26.1% | negative |
| 1.7B k=3 | -21.9% | -30.0% | negative |

The mechanism is measured, not guessed. Draft acceptance was healthy
(median 0.47-0.50 of tokens came from the draft), yet decode still lost.
The draft-plus-verify cycle costs more than a target step that is
already ~9.5 ms on 3B active params. This sharpened the ratio law I
found on the 14B into its real form. **Speculative decoding pays when
the draft's per-step time is a small fraction of the TARGET's per-step
time -- set by active parameters and box speed, not total parameter
count.** A 0.6B draft against ~3B active is a 20% effective ratio, which
fails just like the 15-21% ratios did on the smaller class. The "2% of
total params" framing predicted a win and was wrong. MoE spec-decode is
also NOT the near-tie-flip quality class -- it shows real byte
mismatches and a shifted stop-reason distribution.

### Dense 32B -- promoted, exactly where the law predicts

Target Qwen2.5-Coder-32B (dense, so active = total), baseline 34.7 tps.
Draft = the same 0.5B that won on the 14B (~1.6% ratio). k=2 cleared the
screen (+19.8%). k=3 fell out on the prefill guard.

- **Promoted (`bin/hone-fast-32b`): decode 34.7 -> 41.7 tps (+20.06%)**,
  66/72 pairs positive, sign test p = 7.3e-14, prefill -2.2% (within
  allowance), suite wall -11%.
- **Quality: near-tie-flip class.** 8/12 prompts byte-identical. 4
  diverge deterministically at top-2 gaps of 0.0-0.016 nats, both arms
  byte-stable, stop reasons identical.
- **Replicated** in an independent session: +20.2% (p=7.3e-14), same 4
  divergence positions, same gaps.

The +20% is about 3x smaller than the mini's +75%, and that is the law
being consistent, not disappointing. The dense-32B target step leaves
about 3x less headroom for the draft to hide under.

## T3: three levers beyond greedy short-context decode

The +75% (14B) and +20% (32B) wins are greedy, short-context, single
turn. I then measured three things that matter for real use.

### T3-A: prompt-cache reuse for multi-turn / agent loops -- promoted, lossless

Agent loops re-send a large shared prefix every turn, so their cost is
PREFILL, not decode. I measured time-to-first-token directly. Two arms,
greedy 14B, no draft, to isolate the cache effect. The baseline
re-prefills the whole conversation each turn. The candidate keeps a
persistent KV cache and prefills only the new turn's tokens.

- **prefill -76.6% median** (range -50% to -87%), **~2.15 s saved per
  turn** (a full-prefill turn is ~2.8 s, reuse cuts it to ~0.6 s), 36/36
  pairs favor cache, p = 2.9e-11, replicated across two sessions.
- **Lossless: byte-identical output, 0 mismatches.** Cache reuse is the
  same KV computation, not an approximation -- I verified it
  token-for-token with a small-model probe before trusting it. Decode
  unchanged.
- The saving scales with prefix length. The longer the re-sent context,
  the larger the win.

The deliverable is a usage pattern, not a wrapper, because agent loops
are code.

```python
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)          # persist ACROSS turns
# turn 1: prime with the full initial prompt
for r in stream_generate(model, tok, full_ids, prompt_cache=cache, ...):
    ...
# turn N: feed ONLY the new tokens since the cache's offset --
# never re-send the full transcript. mlx-lm 0.31.3 does not trim a
# common prefix; feeding the whole thing double-processes and garbles.
delta_ids = new_full_templated_ids[len(cached_so_far):]
for r in stream_generate(model, tok, delta_ids, prompt_cache=cache, ...):
    ...
```

The chat template must be prefix-stable (it is for Qwen chatml) for the
delta boundary to be well-defined.

### T3-B: does the speed win survive sampling? -- yes, +58%

Everything above is greedy. Real use often samples, so I re-ran the 14B
champion at temperature 0.7, seed-matched per prompt (I confirmed first
that seeded sampling is reproducible on this stack).

- **decode +58.2%** (72/72 pairs, p = 4.2e-22). The win survives, a bit
  smaller than greedy's +75%.
- **Why smaller.** Draft acceptance falls to a median 0.49 at temp 0.7.
  The speedup scales with acceptance, and acceptance drops as
  temperature rises. Speculation pays across the whole decoding
  spectrum, just less as you get hotter.
- At temperature > 0 the byte-compare gate no longer applies -- spec
  sampling is not sample-path-identical to plain sampling even at a
  matched seed. The right guarantee there is distribution preservation,
  which speculative sampling provides by construction. Each arm is
  per-seed reproducible.

### T3-C: does it hold at long context? -- yes, +60.8%

At ~9k tokens of context the target's attention cost per token rises AND
the draft must itself carry the full context -- either could erase the
win. Greedy 14B, a frozen 4-item suite of ~9.4k-token contexts.

- **decode +60.8%** (24/24 pairs, p = 1.2e-7), byte-lossless (0
  mismatches).
- **The draft's long-context cost is real but small: prefill -4.3%**
  (the 0.5B re-reading 9k tokens), within allowance.
- At long context, prefill dominates the total wall, so suite wall
  improves only -3.1% even with decode +60.8%. The decode speedup pays
  most on long GENERATIONS, less when a big prefix dwarfs a short reply.

### Where this leaves the champion

The 14B speculative-decode configuration holds across every regime I
tested -- greedy short +75%, sampled +58%, long-context +60.8% -- and a
separate, orthogonal lever, prompt-cache reuse, cuts multi-turn prefill
~77% losslessly. The two combine (cache reuse addresses prefill,
speculation addresses decode), and I measured the stack directly. With
the speculative draft active, prompt-cache reuse still cuts prefill
-75.4% losslessly (36/36 paired, p=2.9e-11) while decode is unchanged.
So a multi-turn agent loop gets both wins at once -- the decode speedup
from speculation and the prefill cut from cache reuse.

## Cross-platform coda (NVIDIA A30 / vLLM) -- do the findings travel?

Hone's instrument is MLX on Apple Silicon. As a completing check I re-ran
the two promoted levers on NVIDIA A30 (24GB) GPUs via vLLM, which has
speculative decoding and automatic prefix caching built in.

- **Speculative decoding pays here too, but my quantitative ratio law
  does NOT transfer.** On the A30s a 7B draft won +49% decode on a 32B
  target (a 22% draft/target ratio) and +59% on a 72B target (10%) --
  ratios that FAIL on Apple Silicon, where the draft's prefill tax kills
  the win above ~15%. So the qualitative finding generalizes (drafting
  pays, and a bigger target leaves more room), but the specific thresholds
  are hardware- and framework-specific, not universal constants.
- **Prefix-cache reuse cuts prefill losslessly here too** (-27% on a 32B,
  byte-identical) -- smaller than my -76% only because that run's shared
  prefixes were short. The lever is the same.
- **A portability gotcha worth recording:** vLLM strictly rejects a
  draft/target vocabulary-size mismatch, so my exact champion pair (a
  large target and a small 0.5B draft with a slightly different vocab)
  is not reproducible there at all -- MLX silently allowed it. I also saw
  an int4 + CPU-offload MoE flip a few greedy outputs between runs. A
  later controlled test traced that to batching, not to the quantization
  or the offload. At batch 1 greedy output was bit-identical for every
  weight precision, bf16 included. At batch 8 it flipped on 38-57% of
  outputs, bf16 included, and offload was bit-identical.

The honest completing statement is that the levers are real beyond Apple
Silicon and the tuning constants are not. Measure them per box.

## What I have NOT explored yet

- **The 32B prefill lever on Apple Silicon** -- prompt-cache reuse is
  measured on the 14B here (and confirmed lossless on a 32B via the
  CUDA coda above). A native MLX 32B point wants its own per-box
  calibration on a quiet machine.
- **Draft-model fine-tuning** -- training the 0.5B draft on the target's
  own outputs should raise acceptance and push the ceiling higher. That
  crosses my "no training" non-goal, so it is future work.
- **Compilation and sampler-path overhead** -- listed as knobs, never
  swept. Likely small on this stack, but "likely" is not a measurement.
- **Stack version bumps** -- everything is pinned (mlx 0.32.0 / mlx-lm
  0.31.3). Any upgrade voids and re-runs the calibrations by law.
