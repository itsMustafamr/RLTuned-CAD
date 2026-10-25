"""Tiered, verifiable reward for BRep sequence generation.

    tier 0  output truncated / no end_model() / unparseable          -> 0
    tier 1  parsed: cheap topology checks (no OCC)                    -> 0.05 + 0.15 * s_topo
    tier 2  + wireframe Chamfer to the GT edges (edges only, no faces) -> + 0.20 * s_topo * s_wire     (tiers 1-2 max 0.40)
    tier 3  compiles to a valid closed solid                         -> 0.25 + 0.65 * min(IoU, s_cd) + 0.10 * s_struct

* Ordering: correct valid solid (~1.0) > roughly right valid solid > invalid near-miss
  (<= 0.40) ~ valid but wrong shape (~0.35) > invalid and far off > truncated (0).
  A valid solid of the WRONG shape does not beat a near-miss, so the policy is not
  pushed toward "emit any simple valid box" (checked by scripts/reward_sanity.py).
* min(IoU, s_cd) is ReCAD's anti-gaming trick: both volume overlap and surface
  distance must be good.
* s_struct (face/edge/vertex counts + surface-type mix vs GT) discourages the
  "emit a simple valid box" shortcut. Those counts are also what the text prompt
  states, so this term doubles as a description-consistency check.
* Everything is computed in the canonical [-1,1] frame, no alignment.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Dict

from .checks import structure_signature, structure_similarity, topology_scores
from .seqfmt import extract_sequence, parse

# reward knobs (kept in one place so the paper can quote them)
W = dict(
    base=0.05,          # parsed, complete output
    topo=0.15,          # cheap topology checks
    wire=0.20,          # wireframe similarity (gated by topology)      -> invalid outputs top out at 0.40
    solid=0.25,         # floor for a valid closed solid
    geo=0.65,           # min(IoU, s_cd)
    struct=0.10,        # counts + surface-type mix
)
CD_SCALE = 0.05         # surface Chamfer scale (coordinates in [-1, 1])
WIRE_SCALE = 0.05
WIRE_FLOOR = 0.005
N_PRED_SURF = 20000
N_QUERY = 5000
MAX_BAD_LINE_FRAC = 0.3

FIELDS = ("score", "complete", "parsed", "s_topo", "s_wire", "valid_solid", "iou", "cd", "s_cd", "s_geo",
          "s_struct", "n_faces", "compile_ms", "timeout")


def _empty() -> Dict[str, float]:
    d = {k: 0.0 for k in FIELDS}
    d["cd"] = 9.99
    return d


def cheap_part(text: str):
    """Parent-side work (no OCC): extract, parse, topology checks.
    Returns (seq, partial_result, seq_text); seq is None when the output earns tier 0."""
    out = _empty()
    seq_text = extract_sequence(text)
    seq = parse(seq_text)
    out["complete"] = float(seq.complete)
    out["n_faces"] = float(len(seq.faces))
    bad_frac = seq.n_bad_lines / max(1, seq.n_lines)
    if not seq.complete or not seq.faces or bad_frac > MAX_BAD_LINE_FRAC:
        return None, out, seq_text
    out["parsed"] = 1.0
    t = topology_scores(seq)
    out["s_topo"] = t["s_topo"]
    out["score"] = W["base"] + W["topo"] * t["s_topo"]
    return seq, out, seq_text


@lru_cache(maxsize=512)          # per child process; ~0.5 MB per entry
def _gt(path: str):
    from .gtcache import load_gt_assets
    return load_gt_assets(path)


def full_part(seq_text: str, gt_path: str, partial: Dict[str, float]) -> Dict[str, float]:
    """Child-side work (OCC): wireframe, compile, tessellate, compare. Pure function of its inputs."""
    from . import geometry as G
    from .compiler import compile_text

    out = dict(partial)
    gt = _gt(gt_path)
    seq = parse(seq_text)
    s_topo = out["s_topo"]

    wire = G.sample_edges(seq)
    wcd = G.chamfer(wire, gt["wire"]) if len(wire) else float("inf")
    out["s_wire"] = G.soft_exp(wcd, WIRE_FLOOR, WIRE_SCALE)

    t0 = time.time()
    res = compile_text(seq_text)
    out["compile_ms"] = 1000.0 * (time.time() - t0)
    if not res.is_good_solid:
        out["score"] = W["base"] + W["topo"] * s_topo + W["wire"] * s_topo * out["s_wire"]
        return out

    V, T = G.mesh_of(res.shape)
    pts = G.sample_surface(V, T, N_PRED_SURF, seed=1)
    if len(pts) == 0:
        out["score"] = W["base"] + W["topo"] * s_topo + W["wire"] * s_topo * out["s_wire"]
        return out
    out["valid_solid"] = 1.0
    cd = G.chamfer(pts, gt["surf"], n_query=N_QUERY, seed=2)
    out["cd"] = cd
    out["s_cd"] = G.soft_exp(cd, 1.5 * gt["cd_floor"], CD_SCALE)
    out["iou"] = G.iou(G.solid_occupancy(V, T), gt["occ"])
    out["s_geo"] = min(out["iou"], out["s_cd"])
    out["s_struct"] = structure_similarity(structure_signature(seq), gt["sig"])
    out["score"] = W["solid"] + W["geo"] * out["s_geo"] + W["struct"] * out["s_struct"]
    return out


def score(text: str, gt_path: str) -> Dict[str, float]:
    """In-process scoring (no isolation). Used by eval and tests."""
    seq, partial, seq_text = cheap_part(text)
    if seq is None:
        return partial
    return full_part(seq_text, gt_path, partial)
