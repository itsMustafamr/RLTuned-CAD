#!/bin/bash
#SBATCH --job-name=run06_eval_grpo
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

# STEP 6 - the "SFT + GRPO" row: same protocol as run05 on the newest run04 export.
# Then the side-by-side table:   python scripts/compare.py sft_3b grpo_8k   -> results/comparison.md
# A specific checkpoint:  MODEL=$CKPT/grpo_8k/global_step_30/actor/huggingface NAME=grpo_8k_s30 sbatch ...
set -euo pipefail
cd /work/mech-ai-scratch/mohd7/RLTuned-CAD
source scripts/_common.sh
export RL_EXP=${RL_EXP:-grpo_8k}
export NAME=${NAME:-$RL_EXP}
export MODEL=${MODEL:-$(latest_hf "$CKPT/$RL_EXP")}
source scripts/_eval.sh
python scripts/compare.py sft_3b "$NAME"
echo "### run06 done"
