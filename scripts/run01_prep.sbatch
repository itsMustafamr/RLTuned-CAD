#!/bin/bash
#SBATCH --job-name=run01_prep
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# STEP 1 - data prep + the checks that need no GPU (CPU only, ~10 min on 32 cores).
#
#   1. scripts/prepare_data.py: parse every chunk-9 sequence, collapse exact duplicates,
#      90/3/7 split stratified by simple/complex, SFT parquet (4 prompts per shape), RL
#      parquet in curriculum order, and the GT reward cache (each GT rebuilt by our
#      compiler -> surface samples, 64^3 occupancy, wireframe samples, counts).
#   2. tests/test_verl_reward.py: the verl entry point scores GT ~1.0 through the
#      isolated OCC child, and the fallbacks behave.
#   3. scripts/reward_sanity.py: GT must score 1.000 and every corruption lower, in
#      order of severity -> data/processed/reward_sanity.txt (this table goes in the
#      meeting notes).
#   4. tests/preflight_nova.py: verl's SFT dataset tokenizes our parquet with the real
#      chat template (loss only on the answer), RL prompts fit max_prompt_length, and
#      the token-length estimate used for the splits is checked against the tokenizer.
#   5. scripts/eval.py --oracle: the whole eval path with the GT sequence standing in for
#      the model output -> results/oracle/summary.md must show IoU 1.000, invalid 0.
#
# Requires scripts/setup_nova.sh (env) and scripts/unpack_raw.sh (data/raw) first.
# Gate before run02: "GT rebuilt into valid solids" ~98.7% of attempted, gt reward 1.000,
# preflight prints four OK lines, oracle IoU 1.000.
set -euo pipefail
cd "${REPO:-/work/mech-ai-scratch/mohd7/RLTuned-CAD}"
source scripts/_common.sh

echo "================ step 1/5: prepare_data ================"
python scripts/prepare_data.py --seq-root "$RAW/seq" --ann-root "$RAW/ann" --out "$DATA" --workers "$NCPUS"

echo "================ step 2/5: verl reward entry point ================"
python tests/test_verl_reward.py --manifest "$DATA/manifest.jsonl"

echo "================ step 3/5: reward sanity (80 shapes) ================"
python scripts/reward_sanity.py --manifest "$DATA/manifest.jsonl" --n 80 | tee "$DATA/reward_sanity.txt"

echo "================ step 4/5: preflight (tokenizer, prompt lengths) ================"
python tests/preflight_nova.py --data "$DATA" --model "$BASE_MODEL"

echo "================ step 5/5: oracle eval (GT as the model output) ================"
python scripts/eval.py --name oracle --oracle --test "$DATA/test.jsonl" --workers "$NCPUS"
echo "### run01 done"
