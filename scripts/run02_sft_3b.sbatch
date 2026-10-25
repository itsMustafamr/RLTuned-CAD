#!/bin/bash
#SBATCH --job-name=run02_sft_3b
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

# STEP 2 - stand-in SFT on Qwen2.5-VL-3B-Instruct, text -> BRep sequence, so the RL stage
# is not blocked on Bhavika's checkpoint. Full fine-tune, verl FSDP2 SFT trainer, 2 epochs
# over the 13.5k chunk-9 pairs (3 descriptions per shape + the three combined), 16k
# context, loss on the answer only, LR 1e-5 cosine.
#
# 3B is the "first result" setting (fits one evening); the paper setting is 7B:
#   BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct sbatch --job-name=run02_sft_7b scripts/run02_sft_3b.sbatch
# Works on 4 or 8 GPUs (uses what it sees). Resume-safe: resubmit and it continues.
# Output: $CKPT/sft_qwen2.5-vl-3b-instruct_c9/global_step_N/huggingface  (the GRPO starting point)
# Gate before run03: val loss went down over the 2 epochs; the HF export dir exists.
set -euo pipefail
cd /work/mech-ai-scratch/mohd7/RLTuned-CAD
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
source scripts/_common.sh
source scripts/_sft.sh "$@"
echo "### run02 done"
