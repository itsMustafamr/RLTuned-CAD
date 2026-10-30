#!/bin/bash
# Submit the Nova chain with Slurm dependencies. Clear each gate in docs/GRPO_RUNBOOK_2026-10.md
# before trusting the next stage; the runNN headers say what to look for.
# Usage:  bash scripts/submit_chain.sh <stage>
#   prep    run01 (CPU: data prep, GT cache, reward sanity, preflight)
#   train   run02 SFT -> run03 GRPO smoke -> run04 GRPO 8k, plus run05 eval of SFT after run02
#   eval    run06: eval the newest run04 export and write results/comparison.md
#   all     prep + train chained
#   GPUS=4 bash scripts/submit_chain.sh train    # the 4-GPU A100 nodes (64 cores) instead of the 8-GPU ones
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
G=${GPUS:-8}
stage="${1:-}"

train() {  # $1 = optional dependency job id
  local dep=${1:+--dependency=afterok:$1}
  j2=$(sbatch --parsable $dep --gres=gpu:a100:$G scripts/run02_sft.sbatch)
  j3=$(sbatch --parsable --dependency=afterok:$j2 --gres=gpu:a100:$G scripts/run03_grpo_smoke.sbatch)
  j4=$(sbatch --parsable --dependency=afterok:$j3 --gres=gpu:a100:$G scripts/run04_grpo_8k.sbatch)
  j5=$(sbatch --parsable --dependency=afterok:$j2 scripts/run05_eval_sft.sbatch)
  echo "sft=$j2 smoke=$j3 grpo=$j4 eval_sft=$j5"
}

case "$stage" in
  prep)  sbatch scripts/run01_prep.sbatch ;;
  train) train ;;
  eval)  sbatch scripts/run06_eval_grpo.sbatch ;;
  all)   j1=$(sbatch --parsable scripts/run01_prep.sbatch); echo "prep=$j1"; train "$j1" ;;
  *) echo "usage: bash scripts/submit_chain.sh {prep|train|eval|all}"; exit 2 ;;
esac
squeue -u "$USER"
