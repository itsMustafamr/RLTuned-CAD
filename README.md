# RLTuned-CAD: GRPO for BRepVLM (text to B-rep)

RL side of the BRepVLM project. The model writes a B-rep sequence, a BRep compiler rebuilds it
with OpenCascade, a verifiable geometric reward scores it, and GRPO updates the model
(ReCAD-style, adapted to raw B-rep output). Built on [verl](https://github.com/volcengine/verl) v0.9.1.

![pipeline](figures/pipeline_concept.png)

*Panel (a) is chunk-9 model 00094357. Panel (b)'s rollouts and advantage bars are schematic. Panels (c) and (d)
are measured (run01, `logs/run01_prep_*.log`); the same numbers are in the tables below.*

## Headline results

Held-out chunk-9 test split (255 shapes whose GT rebuilds), greedy decoding, scored by one scorer for
every row (`scripts/eval.py`). Filled in from `results/comparison.md` as runs finish.

| Model | IR (invalid ratio) | IoU 128^3 | IoU_best | CD x1e3 (median) | reward |
|---|---|---|---|---|---|
| oracle (GT sequence as the answer) | 0.000 | 1.000 | 1.000 | 0.067 (sampling floor) | 1.000 |
| SFT only (InternVL3-8B-hf, chunk 9, our stand-in SFT) | run02 then run05: not run yet | | | | |
| SFT + GRPO 8k stage | run04 then run06: not run yet | | | | |

The base model is InternVL3-8B (`OpenGVLab/InternVL3-8B-hf`), the model Bhavika's SFT starts from, so her checkpoint
replaces our stand-in SFT without any other change (`POLICY=`). Qwen2.5-VL-3B-Instruct is only the small model for
pipeline tests; nothing reported comes from it.

## How the RL stage works

1. **Prompt.** The three text descriptions of a part (construction steps, global shape, topology
   relations), ~600-1000 tokens. Image prompts come later (`brepgrpo/prompts.py`).
2. **Policy.** The SFT checkpoint writes 8 B-rep sequences per prompt (vLLM rollouts, 8k tokens in
   the first stage, 16k in the second).
3. **Compiler.** Each sequence is rebuilt into an OpenCascade solid (`brepgrpo/compiler_occ.py`, a
   stand-in until Bhavika's compiler plugs in through `brepgrpo/compiler.py`).
4. **Reward** (`brepgrpo/reward.py`), tiered so broken outputs still get a gradient:

   | Tier | Condition | Reward |
   |---|---|---|
   | 0 | truncated, no `end_model()`, unparseable | 0 |
   | 1 | parses: loops close, edges used twice, references valid (no OCC) | 0.05 + 0.15 · topo |
   | 2 | + edge-sampled ("wireframe") Chamfer to the GT edges | + 0.20 · topo · wire (max 0.40) |
   | 3 | compiles to a valid closed solid | 0.25 + 0.65 · min(IoU, CD-sim) + 0.10 · structure |

   `min(IoU, CD-sim)` is ReCAD's guard against gaming one metric. `structure` compares face / edge /
   vertex counts and the surface-type mix with the target, the numbers the text prompt states. Everything
   is compared in the canonical [-1, 1] frame with no alignment, so position, scale and orientation count.
5. **GRPO** (verl): group-relative advantages over the 8 samples, KL 0.001 to the SFT policy, clip
   0.2 / 0.28, LR 1e-6, curriculum by data order (small planar, planes + cylinders, other surfaces, complex).

## What is verified and what is not (yet)

`run01_prep` reran every CPU check on Nova on 2026-10-26 (log: `logs/run01_prep_*.log`). The numbers in
this section come from that log.

Data (`scripts/prepare_data.py`, chunk 9):

| Stage | Shapes |
|---|---|
| unique shapes (exact duplicates collapsed) | 3,956 |
| with text annotations | 3,856 |
| within the 15k-token limit | 3,736 |
| ground truth rebuilds into a valid closed solid (stand-in compiler) | 3,686 of 3,736 |
| split train / val / test | 3,470 / 116 / 270 |
| SFT samples train / val (4 prompts per shape) | 13,452 / 452 |
| RL prompts, 8k stage / 16k stage | 2,877 / 3,318 |
| test shapes with a rebuildable ground truth (`scripts/eval.py`) | 255 |

Annotation coverage, counted from the chunk-9 tarballs: 4,375 of 4,477 sequences have all three
descriptions and 102 have none (79 simple, 23 complex); no sequence has only some of them. The 102 are
dropped, since text-to-sequence training requires a text prompt.

Checks:

* reward entry point (`tests/test_verl_reward.py`): ground truth is scored through the isolated OCC
  child, and a sample whose ground-truth cache is missing falls back to its topology reward without
  stopping the run
* preflight (`tests/preflight_nova.py`, real Qwen tokenizer): the longest RL prompt is 795 tokens
  (limit 1,280); the token estimate used for the length limits is 1.032 to 1.049 times the real count,
  so it never undercounts; ground truth scores 1.0
* oracle eval (`scripts/eval.py --oracle`, the ground-truth sequence as the model output, 255 test
  shapes): invalid ratio 0.000, IoU 1.000, IoU_best 1.000, CD x1e3 median 0.067 (the sampling floor),
  reward 1.000

Reward sanity (`scripts/reward_sanity.py`, 80 random chunk-9 shapes):

| Variant | n | Reward | ± | Valid solid | IoU | Topology |
|---|---|---|---|---|---|---|
| ground truth | 80 | 1.000 | 0.000 | 1.00 | 1.000 | 1.000 |
| vertices jittered ±2 grid steps | 80 | 0.974 | 0.093 | 0.97 | 0.958 | 1.000 |
| jittered ±8 steps | 80 | 0.516 | 0.327 | 0.34 | 0.320 | 1.000 |
| jittered ±32 steps | 80 | 0.506 | 0.325 | 0.36 | 0.323 | 1.000 |
| jittered ±128 steps | 80 | 0.418 | 0.265 | 0.30 | 0.218 | 1.000 |
| another model's ground truth | 80 | 0.321 | 0.031 | 1.00 | 0.078 | 1.000 |
| a simple box instead | 79 | 0.320 | 0.031 | 1.00 | 0.015 | 1.000 |
| one face deleted (invalid, close) | 80 | 0.385 | 0.071 | 0.01 | 0.012 | 0.934 |
| edge order shuffled inside loops | 80 | 0.996 | 0.009 | 1.00 | 0.994 | 0.938 |
| truncated | 80 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 |
| empty | 80 | 0.000 | 0.000 | 0.00 | 0.000 | 0.000 |

879 reward calls, 57 ms per call on one core. The simple-box row has n = 79 because the one sample that
is itself the box is skipped.

A valid but wrong shape (0.32) does not beat a near miss (0.385), so the policy is not pushed toward
"emit any valid box". Jitter of 8 steps or more breaks validity for most shapes, so those rows sit close
together. The shuffled-loop row shows that our compiler (ShapeFix) tolerates edge order inside a loop.
The BRepVLM draft's Algorithm 2 shows Bhavika's compiler does the same (greedy endpoint chaining, then
ShapeFix_Wire), so with either compiler the solid tier ignores loop order; only the topology check sees it.

Also checked on CPU before run01: every verl override composes against verl v0.9.1's config tree
(`tests/check_verl_configs.py`), and the reward entry point survives verl's by-value transfer into a
separate process.

Not run yet: SFT (run02), the GRPO smoke test (run03), the 8k GRPO stage (run04) and the SFT eval (run05)
are queued on Nova. run03 is a 6-step GRPO run that checks one full step before the main run.

### GPU pipeline test (one A100, Nova desktop, 2026-10-29/30, Qwen2.5-VL-3B as the test model)

Every GPU stage ran once on a small chunk-9 subset (`scripts/make_small_subset.py`: 200 train shapes / 800 SFT
rows, 64 RL prompts, all tier 0-1) with the small model. These numbers check the pipeline; they are not results.
Logs: `logs/*desktop*.log`, `results/small_sft`, `results/small_grpo`. The same test with InternVL3-8B-hf has not
run yet (it needs 4 GPUs, see "Running it on Nova").

| Stage | What ran | Outcome |
|---|---|---|
| SFT | Qwen2.5-VL-3B, 1 epoch, 12 steps, Adam state on CPU | train loss 0.522 -> 0.255, val loss 0.263, 16 min |
| Eval (SFT) | 255 test shapes, greedy, 4,096-token answers | invalid 0.973, complete 0.655, topology pass 0.000, reward 0.080 |
| GRPO smoke | 6 steps x 8 prompts x 8 answers, 2,048-token answers | 6 steps + checkpoint + HF export; reward timeout 0; vLLM and trainer log-probs agree (per-token probability difference mean 0.002, perplexity ratio 1.0002 to 1.0004, KL 2e-4 to 5e-4 over the 6 steps) |
| Eval (after GRPO) | same 255 shapes | invalid 0.992, complete 0.925, reward 0.097 |

The SFT model writes the format with the right face / edge / vertex counts, but its loops do not close and
its edges are not shared between faces, so almost nothing builds. Six GRPO steps raised completion (answers
that end with `end_model()`) and lowered validity, consistent with finishing being the cheapest part of the
reward to collect from this weak start. To watch for that in the real run, `refs_ok`, `wires_closed` and
`manifold` are now logged next to `s_topo`.

## Running it on Nova

Conventions as in VISTA-CAD: `scripts/runNN_name.sbatch` with the explanation in the header,
`scripts/_common.sh` holds the paths, `logs/<job>_<id>.log`, `results/` is tracked.

```bash
cd /work/mech-ai-scratch/mohd7 && git clone <repo url> RLTuned-CAD && cd RLTuned-CAD
bash scripts/setup_nova.sh                 # once, ~20-30 min, needs internet (OnDemand desktop / login node); downloads InternVL3-8B-hf
# copy Bhavika's BRepVLM folder (the two .tar.gz are enough) into data/BRepVLM, then
bash scripts/unpack_raw.sh data/BRepVLM
bash scripts/submit_chain.sh all           # run01 -> run02 -> run03 -> run04, run05 after run02
bash scripts/submit_chain.sh eval          # run06 on the newest GRPO export -> results/comparison.md
```

| Script | GPUs | What |
|---|---|---|
| `run01_prep.sbatch` | 0 | data prep, GT reward cache, reward sanity, preflight, oracle eval |
| `run02_sft.sbatch` | 8 | stand-in SFT, InternVL3-8B-hf, 2 epochs (`BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct` for a pipeline test) |
| `run03_grpo_smoke.sbatch` | 8 | 6 GRPO steps, short answers: proves one full step runs |
| `run04_grpo_8k.sbatch` | 8 | the main GRPO run (stop it any time; run06 takes the newest export) |
| `run05_eval_sft.sbatch` | 4 | SFT-only row |
| `run06_eval_grpo.sbatch` | 4 | SFT + GRPO row and `results/comparison.md` |

Every `.sbatch` is a plain bash script, so inside a Nova Desktop session with GPUs you can run
`bash scripts/run03_grpo_smoke.sbatch` directly; they use however many GPUs they see.
`GPUS=4 bash scripts/submit_chain.sh train` targets the 4-GPU A100 nodes. The runbook with the
verification gate for each stage is `docs/GRPO_RUNBOOK_2026-10.md`.

One-GPU pipeline test without the queue (Nova desktop, 1 A100, 8 cores, about an hour, small model):

```bash
python scripts/make_small_subset.py
export BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct DATA=$PWD/data/processed_small SFT_EXP=sft_small_test EPOCHS=1
bash -c 'source scripts/_common.sh && source scripts/_sft.sh engine.optimizer_offload=True'
bash -c 'source scripts/_common.sh && NAME=small_sft MODEL=$(latest_hf "$CKPT/sft_small_test") MAXTOK=4096 TEST=$PWD/data/processed/test.jsonl && source scripts/_eval.sh'
bash -c 'source scripts/_common.sh && SMOKE=1 EXP=grpo_small_test GPU_UTIL=0.5 POLICY=$(latest_hf "$CKPT/sft_small_test") && source scripts/_grpo.sh actor_rollout_ref.actor.fsdp_config.optimizer_offload=True'
```

The same test with the real base model needs 4 GPUs (8.4B parameters with Adam state do not fit one card):

```bash
DATA=$PWD/data/processed_small SFT_EXP=sft_small_internvl EPOCHS=1 sbatch --gres=gpu:a100:4 --cpus-per-task=32 scripts/run02_sft.sbatch
```

When Bhavika's checkpoint arrives: `POLICY=/path/to/her/hf EXP=grpo_8k_bh sbatch scripts/run04_grpo_8k.sbatch`
(skip our SFT), and `MODEL=/path/to/her/hf NAME=sft_bhavika sbatch scripts/run05_eval_sft.sbatch` for her SFT-only
row. Match `brepgrpo/prompts.py` to her prompt format first. Her checkpoint must be in the native transformers
layout (`InternVLForConditionalGeneration`, as in `OpenGVLab/InternVL3-8B-hf`); the original InternVL layout
(`InternVLChatModel`, remote code) generates in vLLM but its `forward` requires `pixel_values`, which verl does not
pass for text prompts, so it would need a weight conversion first.

### Knobs (environment variables, read by `scripts/_common.sh` and the launchers)

| Variable | Default | Meaning |
|---|---|---|
| `BASE_MODEL` | `OpenGVLab/InternVL3-8B-hf` | SFT starting point (the BRepVLM base); `Qwen/Qwen2.5-VL-3B-Instruct` for pipeline tests |
| `POLICY` | newest SFT export | GRPO starting point |
| `STAGE` | `8k` | `16k` adds the long (complex) shapes; start it from the 8k run's export |
| `N`, `BATCH`, `LR`, `KL` | 8, 64, 1e-6, 0.001 | rollouts per prompt, prompts per step, actor LR, KL coefficient |
| `STEPS` | all (2 epochs) | `trainer.total_training_steps`, e.g. `STEPS=30` for a fixed budget |
| `TP`, `GPU_UTIL` | 1, 0.6 | vLLM tensor parallel size and memory share (vLLM counts the colocated trainer against it; 0.5 for the 1-GPU test) |
| `TQ_UNITS` | number of GPUs | TransferQueue storage units, 1 CPU each (verl's default of 8 needs 9 CPUs) |
| `SFT_SAVE` | `after_each_epoch` | SFT checkpoint interval; an integer N also saves every N steps (use ~50 in scavenger) |
| `TEMPERATURE`, `MAXTOK` | 0.0, 16384 | eval sampling temperature and answer length |
| `BREP_COMPILER_MODULE` | (ours) | module or .py exposing `compile_sequence(text) -> TopoDS_Shape` (Bhavika's) |

## Next steps

1. **GT guidance for hard prompts (ReCAD / LUFFY).** Put the GT sequence into the GRPO group as one of
   the 8 samples on prompts where the policy keeps failing, with LUFFY's r/(r+0.1) weighting. Needs a
   small verl trainer subclass; written once the basic run works on Nova.
2. **16k stage:** `STAGE=16k POLICY=<8k export> EXP=grpo_16k sbatch scripts/run04_grpo_8k.sbatch`.
3. **Image-to-sequence** once renders arrive: an `images` column and `<image>` in the prompt; verl's
   dataset handles the rest.
4. **Bhavika's compiler:** her reconstruction repository (github.com/bhavikab04/BRepVLM) has been shared
   and is not yet integrated. Set `BREP_COMPILER_MODULE`, then rerun run01 to rebuild the GT cache.
5. **InternVL3-8B-hf on a GPU.** Verified so far (transformers 5.9.0, vLLM 0.24.0, verl v0.9.1): the native class
   loads and runs a text-only forward the way verl calls it, vLLM has it in its registry, and the FSDP2 layer list
   the launchers add for it composes (`tests/check_verl_configs.py`) and names the classes that exist in the model
   (`Qwen2DecoderLayer`, `InternVLVisionLayer`; the class itself has no `_no_split_modules`, which is why verl
   needs the list). Not yet run: an SFT step and a GRPO step on a GPU. The 4-GPU small test above is that check:
   watch the preflight (chat template, loss mask, prompt lengths), memory, and seconds per step.

## Running outside Slurm (an AWS instance)

The scripts do not depend on Slurm: every `.sbatch` is a plain bash script and `scripts/_common.sh` reads every
path from an environment variable with a Nova default. On a fresh GPU machine:

1. Pick a machine with 8 x 80 GB GPUs (A100 or H100) and an NVIDIA driver of 580 or newer: verl's lock installs
   torch 2.11 with CUDA 13 wheels, which need an R580+ driver (`scripts/setup_nova.sh` prints the driver and
   warns). Give it a large disk or volume: roughly 15 GB for the verl env, 16 GB for InternVL3-8B-hf, and on the
   order of 100 GB per full checkpoint with optimizer state (`max_actor_ckpt_to_keep=3` keeps three).
2. Put the repo and its sibling verl checkout on that disk and export the paths once per shell:
   ```bash
   export REPO=/data/RLTuned-CAD VERL=/data/verl CKPT=/data/ckpt HF_HOME=/data/hf UV_CACHE_DIR=/data/uv
   cd $REPO && bash scripts/setup_nova.sh && bash scripts/unpack_raw.sh data/BRepVLM
   ```
3. Run the stages as bash scripts, in the same order and with the same gates as on Nova:
   ```bash
   bash scripts/run01_prep.sbatch
   bash scripts/run02_sft.sbatch           # uses every GPU nvidia-smi lists
   bash scripts/run03_grpo_smoke.sbatch && bash scripts/run04_grpo_8k.sbatch
   bash scripts/run05_eval_sft.sbatch && bash scripts/run06_eval_grpo.sbatch
   ```
   Logs still go to `logs/<job>_<id>.log` only under Slurm; outside it, redirect: `bash scripts/run02_sft.sbatch 2>&1 | tee logs/run02_sft_aws.log`.
4. Copy `results/` and the HF exports off the instance before stopping it (an instance's own disk goes with it).

The CPU budget in `scripts/_grpo.sh` assumes `nproc` cores are all for this job, which is true on a dedicated
instance; `REWARD_WORKERS` and `TQ_UNITS` can still be set by hand.

## Troubleshooting

* GRPO never prints a step and Ray warns about pending actors: some actor did not get a CPU. verl holds
  3 CPUs per GPU for training and `TQ_UNITS` + 1 for TransferQueue, and the `remote` reward manager starts
  `reward.num_workers` squared scoring actors (1 CPU each). `_grpo.sh` picks the largest `REWARD_WORKERS`
  that fits (3 on 48 cores + 8 GPUs, 1 on an 8-core desktop); request more cores or lower `TQ_UNITS`.
* `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'mean'` on the first forward pass:
  verl v0.9.1's text-only vision dummy with transformers 5.x. `scripts/setup_nova.sh` patches it (one line,
  as on verl main); rerun setup on an older checkout.
* vLLM `To serve at least one request with the model's max seq len (128000)`: `max_model_len` was unset, so
  vLLM sized for the model's 128k limit. `_grpo.sh` now sets it to prompt + answer length. If it still lacks
  cache memory, raise `GPU_UTIL`: vLLM counts the colocated training process against its share.
* eval `argument --temperature: invalid float value: '/tmp'`: an older `_eval.sh` read `TEMP`, which Nova
  sets to the temp directory; the variable is `TEMPERATURE` now.
* `DataLoader worker ... killed by signal: Killed` after the last GRPO step: seen once at shutdown, after the
  checkpoint and final validation were written. Check `sacct -j <job> -o MaxRSS,ReqMem` if it appears mid-run.
* `prompt_length/clip_ratio` or `response_length/clip_ratio` above 0 with lengths below the limit: verl
  compares each length to the longest one in the batch, not to the configured limit; truncation is real only
  when `response_length/max` equals `RESP`. `perf/cpu_memory_used_gb` is the whole node's memory, not the job's.
* SFT stops with an `input_ids mismatch` message from verl's dataset check: append
  `data.ignore_input_ids_mismatch=True` to the sbatch call.
* vLLM out of memory in GRPO: `TP=2` or `GPU_UTIL=0.5`, or `BATCH=32`.
* `apply_fsdp2` fails with `assert len(fsdp_transformer_layer_cls_to_wrap) > 0` or a `NoneType` error on an
  InternVL checkpoint: the layer list was not passed. The `[sft]` / `[grpo]` launch line prints `wrap=...`;
  `fsdp_wrap_layers` in `scripts/_common.sh` detects InternVL from the model name or the checkpoint's `config.json`.
* `[brep reward] ...` lines in the GRPO log: a sample's OCC child crashed or timed out; it kept its
  topology reward and was flagged `timeout=1`. A steady stream of them means the compiler is unhappy with
  what the policy writes, look at `runs/<exp>/val_generations/`.

## Repo map

```
brepgrpo/     seqfmt (parse), checks (topology), compiler_occ + compiler (OCC), geometry, gtcache,
              reward, verl_reward (verl entry point), prompts, metrics_eval (IoU, IoU_best, CD on faces / edges / vertices)
scripts/      prepare_data, reward_sanity, eval, compare, unpack_raw, setup_nova, submit_chain,
              _common.sh + _sft.sh/_grpo.sh/_eval.sh launchers, run01..run06 sbatch
tests/        test_core (pytest), test_verl_reward, check_verl_configs (every override + the InternVL wrap list), preflight_nova
docs/         GRPO_RUNBOOK_2026-10.md
DATA_REPORT.md  what is in the data and what we ask Bhavika for
figures/      pipeline figure (figures/make_pipeline_figure.py)
```
