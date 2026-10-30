#!/bin/bash
#SBATCH --job-name=run04_grpo_8k
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# STEP 4 - the main GRPO run, 8k stage: 2,877 train prompts whose GT rebuilds and fits in
# ~7.3k tokens, in curriculum order (small planar -> planes+cylinders -> other surfaces ->
# complex), 64 prompts x 8 rollouts per step, 8k-token answers, 2 epochs (~90 steps).
# Reward = brepgrpo/reward.py (parse -> topology -> wireframe -> valid solid: min(IoU, CD) +
# structure), GRPO with KL 0.001 to the SFT policy, clip 0.2/0.28, LR 1e-6.
#
# Checkpoints + HF exports every 15 steps, validation (greedy, 113 val prompts) every 15
# steps -> runs/grpo_8k/val_generations/. Stop it whenever the evening ends: run06 evaluates
# the newest export. Resume-safe (trainer.resume_mode=auto).
#
# Variants:
#   POLICY=/path/to/bhavika/hf EXP=grpo_8k_bh  sbatch scripts/run04_grpo_8k.sbatch    (her checkpoint)
#   STAGE=16k POLICY=$CKPT/grpo_8k/global_step_<last>/actor/huggingface EXP=grpo_16k sbatch ...
# Gate before run06: reward mean and valid_solid rising over the first 30 steps; timeout ~0.
set -euo pipefail
cd "${REPO:-/work/mech-ai-scratch/mohd7/RLTuned-CAD}"
source scripts/_common.sh
export STAGE=${STAGE:-8k} EXP=${EXP:-grpo_8k}
source scripts/_grpo.sh "$@"
echo "### run04 done"
