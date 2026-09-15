# Speculative Decoding Pays Only at Small Active-Parameter Ratios on Apple Silicon: A Pre-Registered Headroom Map for Local LLM Inference

I built Hone to answer one question honestly. How much faster can
local LLM inference get on my own machine, at quality I can verify?

For my daily-driver model, the answer is **+75-77%, replicated.**
Qwen2.5-Coder-14B (4-bit, MLX) goes from 11.9 to 21-22.5 tokens/sec
decode on an M4 Mac mini using tuned speculative decoding -- a 0.5B
draft model, exactly 2 draft tokens, greedy -- with prompt processing
unchanged and output quality characterized down to the individual
tokens that differ (2 prompts in 12 flip at deterministic coin-flip
positions where the model's top two choices sit 0.016 nats apart).

Then I pushed the same machine further. On a second box (M3 Ultra,
96GB) a dense 32B coder gets **+20%** decode from the same technique.
A mixture-of-experts target LOST, which turned out to be the
informative result. It sharpened my rule for when speculative decoding
pays. What matters is the draft's step time relative to the *target's*
step time, not model size. I also measured three levers beyond greedy
short-context decode. **Prompt-cache reuse cuts multi-turn / agent-loop
prefill ~77% losslessly** (about 2.15s saved per turn). The speedup
**survives sampling** (+58% at temperature 0.7) and **long context**
(+60.8% at 9k tokens). Full numbers in `RESULTS.md`, and the full
write-up is in [`paper/hone.md`](paper/hone.md).

![T1 decode speed, stock vs speculative decoding, per prompt](figures/fig1_t1_decode.png)

The speedup is not the interesting part -- speculative decoding is
public knowledge. The interesting part is the measurement machine
around it.

- **Margins were registered before any comparison was judged.** I ran
  identical-vs-identical (A/A') calibrations first, measured the paired
  noise floor, and froze the promotion margins in `PROTOCOL.md` as
  append-only amendments written before the results they govern.
- **Every gate is mechanical.** Speed (paired sign test + registered
  margin), prompt-processing guard, quality parity, run integrity
  (token recounts, wall-clock sanity, contamination checks), and a
  different-day replication -- no eyeballing anywhere.
- **The protocol caught my own rules being wrong three times** -- a
  contamination check that flagged a legitimate +79% win as noise, a
  memory-pressure guard that measured cache warmth instead of risk, and
  a sanity rule that compared single-pair extremes against
  median-scale margins. Each fix shipped with a canary test that failed
  once against the unpatched code, and was backtested on data already
  in hand before being registered.
- **Negative results are results.** The full headroom map in
  `RESULTS.md` includes what did NOT work: larger drafts lose to their
  own prompt-processing cost, drafting more than ~3 tokens ahead loses
  outright, KV-cache quantization costs decode speed here, and the
  planned self-tuning search loop was cancelled when the measured
  search space turned out to be one-dimensional -- per an exit rule I
  registered before running the sweep.

## Use it

```
bin/hone-fast "your prompt"
```

runs the promoted 14B configuration via `mlx-lm`, and
`bin/hone-fast-32b` runs the promoted 32B configuration.

## Reproduce

- **Environment.** `env.lock` pins the full stack (mlx 0.32.0, mlx-lm
  0.31.3, Python 3.14.3). Any stack version bump voids and re-runs the
  calibrations by law. Model weights are content-hashed, and suite JSONs
  are hashed into every run header, so any mismatch voids the run.
- **Measure and judge.** `scripts/measure.py` runs paired, interleaved,
  guarded A/B measurements. `scripts/verdict.py` judges any run against
  the registered gates. Raw run evidence for every headline number is in
  `evidence/`.
- **Tests.** `python -m pytest` runs the suite in `tests/`: gate logic,
  box-safety guards, the measurement harness, verdict machinery, and
  paired statistics.
- **Figures.** `python scripts/make_figures.py` (needs matplotlib and
  numpy) rebuilds every chart in `figures/` from `evidence/` with the
  same paired statistics the promotion gates use. No plotted value is
  typed by hand, and the script stops if a recomputed headline does not
  match `RESULTS.md`. The bootstrap confidence intervals in the figure
  titles and error bars are computed by this script.
- **Paper.** `bash scripts/render_paper.sh` (needs pandoc and Chrome)
  renders `paper/hone.md` to `paper/hone.pdf` with the figures inline.

- `DESIGN.md` -- what the machine is and the rules it runs under
- `PROTOCOL.md` -- pre-registered measurement laws and amendments
- `RESULTS.md` -- the full headroom map, both boxes, the T3 levers, the
  promotion record, and what I have not explored yet
- `paper/hone.md` -- the full write-up: method, protocol, results,
  negative results, and threats to validity
- `paper/hone.pdf` -- the same paper rendered with figures inline
- `figures/` -- every chart, generated from `evidence/` (see Reproduce)
