# Shared settings for every job. Sourced by scripts/run*.sbatch and the _sft/_grpo/_eval launchers.
# Every path is an environment variable with a Nova default, so the same scripts run on another machine
# (an AWS instance, say) by exporting REPO, VERL, CKPT and HF_HOME first; see "Running outside Slurm" in the README.

export REPO=${REPO:-/work/mech-ai-scratch/mohd7/RLTuned-CAD}
export VERL=${VERL:-/work/mech-ai-scratch/mohd7/verl}        # verl v0.9.1 sibling checkout + its .venv (scripts/setup_nova.sh)
export RAW=${RAW:-$REPO/data/raw}                             # unpacked chunks (scripts/unpack_raw.sh)
export DATA=${DATA:-$REPO/data/processed}                     # output of scripts/prepare_data.py
export CKPT=${CKPT:-/work/mech-ai-scratch/mohd7/ckpt/rltuned-cad}
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/mohd7/.cache/huggingface}
# SFT starting point. InternVL3-8B is the BRepVLM base model (Bhavika's SFT starts from it), in the native
# transformers layout (-hf) that verl trains directly. Qwen/Qwen2.5-VL-3B-Instruct is the small model for
# pipeline tests; nothing reported comes from it.
export BASE_MODEL=${BASE_MODEL:-OpenGVLab/InternVL3-8B-hf}
MODEL_TAG=$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')
export MODEL_TAG
export SFT_EXP=${SFT_EXP:-sft_${MODEL_TAG}_c9}                 # e.g. sft_internvl3-8b-hf_c9

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
# the padded SDPA path: identical math, slower. The launchers pass these overrides.
# On the padded path every micro-batch is padded to its longest sample and verl materializes the full
# (micro-batch x longest x 151,936-word vocab) logits, so token-budget micro-batches that mix short and
# long samples need several times their real memory. DYNAMIC_BSZ=False: one sample per micro-batch, no padding.
if python -c 'import flash_attn' 2>/dev/null; then
  export REMOVE_PADDING=True ATTN_IMPL=flash_attention_2 DYNAMIC_BSZ=True
else
  export REMOVE_PADDING=False ATTN_IMPL=sdpa DYNAMIC_BSZ=False
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

# verl's FSDP2 wrapper shards a model at the layer classes the model lists in `_no_split_modules`
# (verl/utils/fsdp_utils.py, apply_fsdp2) and stops if the list is empty. transformers' native InternVL
# class lists none, so for InternVL checkpoints the layers are named explicitly: the Qwen2 decoder layers of
# its language model and the vision encoder layers. Prints nothing for other models (Qwen2.5-VL has its own
# list), and the launchers then pass no override. $1 = HF model id or a local checkpoint directory.
fsdp_wrap_layers() {
  local probe="$1"
  [ -f "$1/config.json" ] && probe=$(cat "$1/config.json")
  case "${probe,,}" in *internvl*) echo "[Qwen2DecoderLayer,InternVLVisionLayer]" ;; esac
}

echo "[common] host=$(hostname) gpus=$NGPUS cpus=$NCPUS model=$BASE_MODEL sft_exp=$SFT_EXP attn=$ATTN_IMPL remove_padding=$REMOVE_PADDING dynamic_bsz=$DYNAMIC_BSZ"
