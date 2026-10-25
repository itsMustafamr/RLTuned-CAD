#!/bin/bash
#SBATCH --job-name=run05_eval_sft
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# STEP 5 - the "SFT only" row: greedy decoding on the 255 held-out test shapes with the
# run02 export, scored by the training reward and the paper metrics (invalid ratio, IoU at
# 128^3, IoU_best over 24 axis rotations, Chamfer). One vLLM process per GPU, then merge.
# Output: results/sft_3b/{summary.md,summary.json,predictions*.jsonl}
# Can run as soon as run02 finishes (in parallel with run03/run04).
set -euo pipefail
cd /work/mech-ai-scratch/mohd7/RLTuned-CAD
source scripts/_common.sh
export NAME=${NAME:-sft_3b}
export MODEL=${MODEL:-$(latest_hf "$CKPT/$SFT_EXP")}
source scripts/_eval.sh
echo "### run05 done"
