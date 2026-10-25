#!/usr/bin/env bash
# ONE-TIME environment setup on Nova. Run where you have internet (your OnDemand desktop
# or a login node), NOT inside an sbatch job:
#
#     cd /work/mech-ai-scratch/mohd7 && git clone <repo url> RLTuned-CAD && cd RLTuned-CAD
#     bash scripts/setup_nova.sh
#
# What it does (idempotent, safe to re-run, ~20-30 min, mostly downloads):
#   1. installs uv if missing
#   2. clones verl v0.9.1 as a SIBLING checkout (kept pristine):
#        /work/mech-ai-scratch/mohd7/verl
#      and builds its LOCKED env there with `uv sync --extra fsdp --extra vllm`, which is
#      verl's own documented combo: torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.9.0,
#      flash-attn 2.8.3. Nova's A100 driver is 580.x = CUDA 13, so the cu130 wheels fit.
#      verl's lock file supports Python 3.12 only, so the venv is pinned to 3.12 (uv
#      downloads it if the node has none; conda does not need to be active, deactivate it).
#      flash-attn is NOT installed: the lock pins a prebuilt wheel from verl's wheelhouse
#      (flash-attention-v2.8.3, torch 2.11 / cu130) that the wheelhouse has since removed
#      (404; it now only serves torch-2.13 / cu132 builds, which do not match this torch).
#      The launchers detect the missing package and run verl with
#      use_remove_padding=False + attn_implementation=sdpa: same math, slower.
#      To get flash-attn back later: build it against this torch with a CUDA 13 toolkit
#      (`uv pip install --python $VERL/.venv/bin/python --no-build-isolation flash-attn==2.8.3`, ~30+ min).
#   3. adds our few extra deps to that venv (OpenCascade via cadquery-ocp 8.x, pebble,
#      regex, pyarrow) - requirements-extra.txt
#   4. import check (CPU only)
#   5. downloads the base model into $HF_HOME (compute nodes have no internet)
# Then: bash scripts/unpack_raw.sh data/BRepVLM  and  sbatch scripts/run01_prep.sbatch
set -euo pipefail

REPO=${REPO:-/work/mech-ai-scratch/mohd7/RLTuned-CAD}
VERL=${VERL:-/work/mech-ai-scratch/mohd7/verl}
export HF_HOME=${HF_HOME:-/work/mech-ai-scratch/mohd7/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/work/mech-ai-scratch/mohd7/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/work/mech-ai-scratch/mohd7/.cache/uv-python}   # if uv has to download a Python 3.12, keep it off the home quota
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
cd "$REPO"
source ~/.bashrc >/dev/null 2>&1 || true
export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH

echo "== 1/5 uv =="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== 2/5 verl v0.9.1 sibling checkout + locked env: $VERL =="
if [ ! -d "$VERL/.git" ]; then
    git clone --depth 1 --branch v0.9.1 https://github.com/volcengine/verl.git "$VERL"
else
    echo "   exists: $VERL ($(git -C "$VERL" describe --tags --always))"
fi
if [ -x "$VERL/.venv/bin/python" ] && ! "$VERL/.venv/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo "   existing .venv is not Python 3.12, rebuilding it"
    rm -rf "$VERL/.venv"
fi
(cd "$VERL" && uv sync --frozen --python 3.12 --extra fsdp --extra vllm --no-install-package flash-attn)

echo "== 3/5 our extras into the same venv =="
uv pip install --python "$VERL/.venv/bin/python" -r requirements-extra.txt

echo "== 4/5 import check =="
"$VERL/.venv/bin/python" - <<'PY'
import torch, vllm, transformers, verl, OCP, pebble, sys
print(f"   python {sys.version.split()[0]} | torch {torch.__version__} (cuda {torch.version.cuda}) | vllm {vllm.__version__} | "
      f"transformers {transformers.__version__} | verl {getattr(verl, '__version__', '?')} | OCP ok")
try:
    import flash_attn
    print("   flash-attn", flash_attn.__version__, "(remove padding on)")
except ImportError:
    print("   flash-attn not installed -> launchers use use_remove_padding=False, attn_implementation=sdpa")
PY

echo "== 5/5 base model -> $HF_HOME =="
"$VERL/.venv/bin/hf" download "$BASE_MODEL" >/dev/null && echo "   downloaded $BASE_MODEL"
echo
echo "setup done. Next: bash scripts/unpack_raw.sh data/BRepVLM   then   sbatch scripts/run01_prep.sbatch"
