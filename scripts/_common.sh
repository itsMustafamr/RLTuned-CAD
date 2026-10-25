# Shared settings for every Nova job. Sourced by scripts/run*.sbatch and the _sft/_grpo/_eval launchers.
# Edit the paths here once; everything else reads them.

export REPO=${REPO:-/work/mech-ai-scratch/mohd7/RLTuned-CAD}
export VERL=${VERL:-/work/mech-ai-scratch/mohd7/verl}        # verl v0.9.1 sibling checkout + its .venv (scripts/setup_nova.sh)
export RAW=${RAW:-$REPO/data/raw}                             # unpacked chunks (scripts/unpack_raw.sh)
export DATA=${DATA:-$REPO/data/processed}                     # output of scripts/prepare_data.py
export CKPT=${CKPT:-/work/mech-ai-scratch/mohd7/ckpt/rltuned-cad}
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/mohd7/.cache/huggingface}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}  # SFT starting point; 7B is the paper setting
MODEL_TAG=$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')
export SFT_EXP=${SFT_EXP:-sft_${MODEL_TAG}_c9}                 # e.g. sft_qwen2.5-vl-3b-instruct_c9

cd "$REPO"
source ~/.bashrc >/dev/null 2>&1 || true
export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH
PY="$VERL/.venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY missing - run scripts/setup_nova.sh first"; exit 1; }
# shellcheck disable=SC1091
set +u; source "$VERL/.venv/bin/activate"; set -u
export PYTHONPATH=$REPO
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}   # avoid JIT kernels that need nvcc on the node
export WANDB_MODE=${WANDB_MODE:-offline}                                 # "online" after `wandb login`
export LOGGER=${LOGGER:-'["console","tensorboard"]'}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$REPO/runs/tensorboard}

# flash-attn could not be installed from verl's lock (see scripts/setup_nova.sh); without it verl runs
# the padded SDPA path: identical math, slower. The launchers pass these two overrides.
if python -c 'import flash_attn' 2>/dev/null; then
  export REMOVE_PADDING=True ATTN_IMPL=flash_attention_2
else
  export REMOVE_PADDING=False ATTN_IMPL=sdpa
fi

NGPUS=$( (nvidia-smi -L 2>/dev/null || true) | wc -l)
NCPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export NGPUS NCPUS
mkdir -p logs results runs "$CKPT" "$HF_HOME"

# newest HF export under a verl checkpoint dir (SFT: global_step_N/huggingface, RL: global_step_N/actor/huggingface)
latest_hf() {
  local d
  d=$(ls -d "$1"/global_step_* 2>/dev/null | sort -V | tail -1)
  if [ -d "$d/actor/huggingface" ]; then echo "$d/actor/huggingface"; elif [ -d "$d/huggingface" ]; then echo "$d/huggingface"; fi
}

echo "[common] host=$(hostname) gpus=$NGPUS cpus=$NCPUS model=$BASE_MODEL sft_exp=$SFT_EXP attn=$ATTN_IMPL remove_padding=$REMOVE_PADDING"
