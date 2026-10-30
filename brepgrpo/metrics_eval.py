"""Paper-style evaluation metrics (denser than the training reward).

  iou128      solid IoU on a 128^3 grid, canonical frame (no alignment)
  iou_best    best IoU over the 24 axis-aligned rotations (ReCAD / CAD-Coder style IoU_best), 64^3
  cd_l2sq     Chamfer distance between the two solids' surfaces: mean of squared NN distances, 30k points
  cd_l2       same with plain distances                                    (these two are the paper's CD-Faces)
  cd_vert     Chamfer distance between the two sequences' vertex sets        (CD-Vert.)
  cd_edge     Chamfer distance between points sampled along the two sequences' edges (CD-Edges)

The BRepVLM draft reports CD on vertices, edges and faces (Tables 3 and 4). cd_vert and cd_edge are computed
from the sequences, so they exist for any output that parses, not only for valid solids; the eval summary
still reports them over valid solids so every CD column covers the same shapes. All distances are plain
(not squared) except cd_l2sq, and all are in the canonical [-1, 1] frame.
GT geometry is the GT sequence rebuilt by the same compiler.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as G
from .compiler import compile_text
from .seqfmt import extract_sequence, parse

N_PTS = 30000
N_EDGE = 3000


@lru_cache(maxsize=512)
def _gt_mesh(seq_path: str):
    res = compile_text(open(seq_path).read())
    if not res.is_good_solid:
        return None
    return G.mesh_of(res.shape)


@lru_cache(maxsize=512)
def _gt_points(seq_path: str):
    """GT vertices and edge samples, from the sequence (no solid needed)."""
    seq = parse(open(seq_path).read())
    return _verts(seq), G.sample_edges(seq, N_EDGE)


def _verts(seq) -> np.ndarray:
    return np.asarray(list(seq.verts.values()), np.float32).reshape(-1, 3)


def _cd(a, b):
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return 0.5 * (da.mean() + db.mean()), 0.5 * ((da ** 2).mean() + (db ** 2).mean())


def eval_metrics(text: str, gt_seq_path: str) -> dict:
    nan = float("nan")
    out = dict(iou128=0.0, iou_best=0.0, cd_l2=nan, cd_l2sq=nan, cd_vert=nan, cd_edge=nan)
    seq_text = extract_sequence(text)
    seq = parse(seq_text)
    gt_v, gt_e = _gt_points(gt_seq_path)
    pv, pe = _verts(seq), G.sample_edges(seq, N_EDGE)
    if len(pv) and len(gt_v):
        out["cd_vert"] = float(_cd(pv, gt_v)[0])
    if len(pe) and len(gt_e):
        out["cd_edge"] = float(_cd(pe, gt_e)[0])

    gt = _gt_mesh(gt_seq_path)
    if gt is None:
        return out
    res = compile_text(seq_text)
    if not res.is_good_solid:
        return out
    V, T = G.mesh_of(res.shape)
    Vg, Tg = gt
    p = G.sample_surface(V, T, N_PTS, seed=11)
    q = G.sample_surface(Vg, Tg, N_PTS, seed=12)
    if len(p) == 0 or len(q) == 0:
        return out
    cd, cd2 = _cd(p, q)
    out["cd_l2"], out["cd_l2sq"] = float(cd), float(cd2)
    out["iou128"] = G.iou(G.solid_occupancy(V, T, 128), G.solid_occupancy(Vg, Tg, 128))
    gocc = G.solid_occupancy(Vg, Tg, 64)
    out["iou_best"] = max(G.iou(G.solid_occupancy(V @ R.T, T, 64), gocc) for R in G.cube_rotations())
    return out
