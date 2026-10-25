"""Prompt templates (text -> BRep sequence). One place, so SFT, RL and eval match."""
from __future__ import annotations

from typing import Dict, List, Tuple

SYSTEM_PROMPT = (
    "You are a CAD engineer. Write the boundary representation (B-rep) of the described part "
    "in the BRepVLM v3 sequence format. The part sits in a [-1, 1] box (longest side = 2 units); "
    "all numbers are 12-bit integers on that grid (0..4094, 2047 = 0.0). List vertices v(...), then "
    "each face (surface + wires with signed 1-based edge references), then edges. "
    "Output only the sequence, starting with model(3) and ending with end_model()."
)

_LABELS = dict(construction_steps="Construction", global_shape="Shape", topology_relations="Topology")


def sft_user_prompts(ann: Dict[str, str]) -> List[Tuple[str, str]]:
    """Single descriptions (abstract-ish) + the combined one (precise), as in ReCAD's SFT mix."""
    out = [(k, ann[k]) for k in ("construction_steps", "global_shape", "topology_relations") if k in ann]
    if len(ann) > 1:
        out.append(("combined", rl_user_prompt(ann)))
    return out


def rl_user_prompt(ann: Dict[str, str]) -> str:
    """RL uses the most specific prompt (ReCAD: detailed descriptions only in RL)."""
    parts = [f"{_LABELS[k]}: {ann[k]}" for k in ("construction_steps", "global_shape", "topology_relations") if k in ann]
    return "\n\n".join(parts)
