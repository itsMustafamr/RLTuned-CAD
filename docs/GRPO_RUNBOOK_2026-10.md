# RLTuned-CAD: first GRPO run on Nova (runbook, 2026-10)

**Goal:** one honest "SFT only vs SFT + GRPO" comparison on chunk 9 with our stand-in compiler and
reward, plus the evidence that every piece of the pipeline does what it claims. Qwen2.5-VL-3B is the
first-result setting; 7B is the paper setting and only changes `BASE_MODEL`.

**Nova root:** `/work/mech-ai-scratch/mohd7/RLTuned-CAD` · verl lives in the sibling checkout
`/work/mech-ai-scratch/mohd7/verl` with its own `.venv` (`scripts/setup_nova.sh`) · every job logs to
`logs/<job>_<id>.log` · SFT and GRPO are resume-safe (resubmit the same job).

Run the stages **in order** and clear each gate before trusting the next one. Time figures are
estimates from model size and token counts, not measurements; the first run measures them.

---

## Stage 0: environment (once, ~20-30 min, needs internet)

In the OnDemand desktop or on a login node, not in an sbatch job:

```bash
cd /work/mech-ai-scratch/mohd7 && git clone <repo url> RLTuned-CAD && cd RLTuned-CAD
bash scripts/setup_nova.sh
mkdir -p data/BRepVLM && cp -r /path/to/BRepVLM/{Sequences_chunk9,Text_Annotations_chunk9} data/BRepVLM/
bash scripts/unpack_raw.sh data/BRepVLM
```

Gate: the import check prints `torch 2.11.0 (cuda 13.0) | vllm 0.24.0 | transformers 5.9.0 | ... | OCP ok`
and `unpack_raw.sh` reports 4477 sequences / 5995 annotated models.

If `uv sync` fails on a wheel download, re-run the script; it is idempotent. If Nova's default python is
older than 3.10, uv downloads its own 3.12 (needs internet, which the desktop has).

## Stage 1: data prep and CPU checks (`run01_prep`, ~10 min on 32 cores)

```bash
sbatch scripts/run01_prep.sbatch
```

Gate (in the log):

| Line | Expect |
|---|---|
| `unique shapes with annotations` | 3856 |
| `GT rebuilt into valid solids` | 3686/3736 attempted |
| `rl_train_8k` | 2877 prompts |
| `GT through isolated child: OK` | yes |
| reward sanity `gt` row | 1.000, every other row lower, `truncated` 0.000 |
| preflight `[1]..[4]` | all OK; `[4]` must say "never undercounts" |
| oracle `all` row | invalid_ratio 0.000, iou_mean 1.000 |

`data/processed/reward_sanity.txt` and `results/oracle/summary.md` are the evidence for this stage.

## Stage 2: stand-in SFT (`run02_sft_3b`, estimate 30-60 min on 8 A100 for 3B)

```bash
sbatch scripts/run02_sft_3b.sbatch            # or: bash scripts/submit_chain.sh train  (chains 2 -> 3 -> 4, and 5)
```

Gate: `val/loss` decreases across the 2 epochs; `latest_hf` prints
`$CKPT/sft_qwen2.5-vl-3b-instruct_c9/global_step_<N>/huggingface`. If the dataset check stops the run with
`input_ids mismatch`, resubmit with `data.ignore_input_ids_mismatch=True` appended.

## Stage 3: GRPO smoke test (`run03_grpo_smoke`, estimate 20-30 min)

6 steps, 8 prompts x 8 rollouts, 2k-token answers. Nothing here is a result.

Gate: 6 step lines; `reward_extra_info` contains `valid_solid`, `s_topo`, `timeout`; mean `timeout` ~0;
no Ray warning about actors waiting for CPUs; `$CKPT/grpo_smoke/global_step_6/actor/huggingface` exists.
Note the seconds per step: the main run has ~8x more rollout tokens per step.

## Stage 4: GRPO 8k stage (`run04_grpo_8k`, 2 epochs ~90 steps; estimate 2-4 min per step for 3B)

```bash
sbatch scripts/run04_grpo_8k.sbatch           # STEPS=30 sbatch ... for a fixed budget
```

Watch `critic/score/mean` (the reward) and `reward_extra_info/valid_solid` in the log or TensorBoard
(`runs/tensorboard`). Checkpoints and HF exports every 15 steps. Validation generations every 15 steps in
`runs/grpo_8k/val_generations/` are the before/after examples.

Gate: reward and valid_solid trend up over the first 30 steps; `timeout` stays ~0; response length does
not collapse to the maximum (that would mean the policy learned to ramble; raise `KL` to 0.005).

## Stage 5 and 6: evaluation (`run05_eval_sft`, `run06_eval_grpo`, estimate 20-40 min each on 4 GPUs)

```bash
sbatch scripts/run05_eval_sft.sbatch          # as soon as run02 is done
sbatch scripts/run06_eval_grpo.sbatch         # newest run04 export; MODEL=... NAME=... for a specific step
python scripts/compare.py sft_3b grpo_8k      # -> results/comparison.md
```

Gate: `results/<name>/summary.md` has n = 255 in the `all` row for both; the comparison table has both rows.

---

## If the evening runs out

The report order of preference, each line only with numbers that exist in a log:

1. Stage 1 evidence (compiler rate, reward ladder, oracle eval) + SFT baseline row + GRPO curve for the
   steps that finished + eval of the newest export.
2. Same without the GRPO eval: curve only.
3. Same without GRPO: Stage 1 evidence + SFT row + the smoke test showing one full step works.

Anything below 3 is "pipeline built and verified on CPU, first GPU run pending", which is also a true
sentence.

## Known limits to state up front

* Stand-in compiler and stand-in SFT: both get replaced by Bhavika's (`BREP_COMPILER_MODULE`, `POLICY=`).
* Both compilers tolerate edge order inside loops (ours via ShapeFix, hers via endpoint chaining +
  ShapeFix_Wire per the draft's Algorithm 2), so the solid tier of the reward ignores loop order.
* One chunk: 3,470 training shapes. SFT quality caps what RL can start from.
* GT guidance (ReCAD / LUFFY) and the 16k stage are not in this run.
