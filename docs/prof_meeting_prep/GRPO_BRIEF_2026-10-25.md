# BRepVLM RL stage — meeting brief (prepared 2026-10-25)

*Speaking notes. Every number comes from a file in this repo (`data/processed/reward_sanity.txt`,
`results/*/summary.md`, `logs/`). Fields marked `[run0N]` are filled from that job's output.*

## The 30-second version

We now have a working ReCAD-style GRPO pipeline for BRepVLM on Nova. The model writes the raw B-rep
sequence, our OpenCascade compiler rebuilds it, a tiered verifiable reward scores it (does it parse,
does the topology close, do the edges match, does it build, does the solid match), and verl's GRPO
updates the policy. Every CPU-side piece is verified on chunk 9: the compiler rebuilds 98.7% of the
ground truth, the ground truth scores exactly 1.0, every corruption scores lower, and the eval path
returns a perfect score for a perfect answer. First GPU results: `[run05/run06]`.

## The one table

Held-out chunk-9 test split, 255 shapes, greedy decoding, one scorer for every row.

| Model | IR (invalid)↓ | IoU 128^3↑ | IoU_best↑ | CD x1e3 (median)↓ | reward↑ |
|---|---|---|---|---|---|
| oracle (GT as the answer) | 0.000 | 1.000 | 1.000 | ~0.07 (sampling floor) | 1.000 |
| SFT only, Qwen2.5-VL-3B | `[run05]` | | | | |
| SFT + GRPO 8k, `[N]` steps | `[run06]` | | | | |

GRPO curve (reward mean and valid-solid rate per step): `[run04 log / TensorBoard screenshot]`.
Before / after on the same prompts: `runs/grpo_8k/val_generations/` step 0 vs step `[N]`.

## What is verified (CPU, reproducible with `sbatch scripts/run01_prep.sbatch`)

| Check | Result |
|---|---|
| sequences parsed | 4,477, zero bad lines; 3,956 unique shapes, 3,856 with annotations |
| stand-in compiler | 3,805 / 3,856 GT rebuild into valid closed solids (98.7%); volumes match annotations, median error 0.1% |
| reward on GT | 1.000 on all 80 sampled shapes |
| reward ladder | jitter ±2 steps 0.974 · ±8 0.516 · ±128 0.418 · one face deleted 0.385 · wrong part 0.321 · box 0.320 · truncated 0 |
| eval path (oracle) | invalid 0.000, IoU 1.000, IoU_best 1.000 |
| verl configs | 94 overrides compose against verl v0.9.1; reward entry point survives verl's process transfer |

## Decisions I made and why (the ones he may ask about)

1. **Tiered reward instead of ReCAD's.** ReCAD's printed weights are 0.9 format / 0.1 geometry, which is
   mostly a format reward for their short Python programs. Our outputs are 2k-10k-token raw B-reps that
   break easily, so broken outputs need a gradient too: parse (0.05) -> topology (0.15) -> wireframe
   match (0.20) -> valid solid (0.25 + 0.65 min(IoU, CD) + 0.10 structure). A valid wrong shape scores
   below an almost-right broken one, so the policy is not pushed toward "emit any valid box".
2. **No alignment in the reward.** Everything is compared in the canonical [-1, 1] frame; an alignment-
   invariant reward would let the policy ignore position, scale and orientation. IoU_best (24 axis
   rotations, as ReCAD reports) is evaluation-only.
3. **Curriculum by data order, not by code.** Train prompts sorted small planar -> planes + cylinders ->
   other surfaces -> complex; `data.shuffle=False`. Second stage raises the answer limit to 16k.
4. **OCC runs in an isolated child with a timeout.** OpenCascade can hang or segfault on a malformed
   sequence; in verl's remote reward manager that would stall the step or kill the Ray actor. A timed-out
   sample keeps its topology reward and is flagged; the step continues.
5. **3B for the first run.** 7B is the paper setting and is one variable (`BASE_MODEL`).
6. **Stand-ins.** Our compiler and our SFT exist only because hers have not arrived; both plug in
   (`BREP_COMPILER_MODULE`, `POLICY=`). The GT cache is rebuilt with whichever compiler scores the policy,
   so a perfect answer scores 1.0 under either.

## Known limits

* One chunk, 3,470 training shapes: the SFT baseline caps what RL starts from.
* Our compiler tolerates edge order inside a loop (ShapeFix); Bhavika's may not. Swap it in early.
* Not in this run: GT guidance on hard prompts (ReCAD / LUFFY, needs a small verl trainer subclass),
  the 16k stage, image prompts.

## Asks

1. Bhavika's compiler / reconstruction code and her SFT checkpoint (base model, prompt format, whether the
   target includes the header lines).
2. Metadata JSONs and image renders for image-to-sequence.
3. Format: 85% of `bspline` edges are straight two-pole segments (7% of tokens); `centroid` fields are
   7.7% of tokens and unused by our compiler. Shorter sequences = cheaper RL.
4. More chunks once the first numbers look right (3 to 5 more).
