#!/bin/bash
#SBATCH --job-name=run03_grpo_smoke
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# STEP 3 - GRPO plumbing check: 6 steps, batch 8 prompts x 8 rollouts, 2k-token answers,
# starting from the run02 export. Nothing here is a result; it only proves that one full
# step runs on Nova: vLLM rollout -> remote reward workers (OCC in isolated children) ->
# advantages -> FSDP update -> checkpoint + HF export -> validation generations.
#
# Gate before run04 (read logs/run03_grpo_smoke_<job>.log):
#   * 6 "step" lines with critic/score/mean (reward) and reward_extra_info keys
#     (valid_solid, s_topo, timeout ...) in the metrics
#   * timeout mean ~0 (reward children are not dying)
#   * no "switching", "ProcessExpired" or Ray scheduling warnings about CPUs
#   * $CKPT/grpo_smoke/global_step_6/actor/huggingface exists
set -euo pipefail
cd /work/mech-ai-scratch/mohd7/RLTuned-CAD
source scripts/_common.sh
export SMOKE=1 EXP=grpo_smoke
source scripts/_grpo.sh "$@"
echo "### run03 done"
