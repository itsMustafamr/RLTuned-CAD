"""Paper-style evaluation metrics (denser than the training reward).

  iou128      solid IoU on a 128^3 grid, canonical frame (no alignment)
  iou_best    best IoU over the 24 axis-aligned rotations (ReCAD / CAD-Coder style IoU_best), 64^3
  cd_l2sq     Chamfer distance, mean of squared NN distances (both directions summed / 2), 30k pts
  cd_l2       same with plain distances
GT geometry is the GT sequence rebuilt by the same compiler.
"""
from __future__ import annotations

from functools import lru_cache

from scipy.spatial import cKDTree

from . import geometry as G
from .compiler import compile_text
from .seqfmt import extract_sequence

N_PTS = 30000


@lru_cache(maxsize=512)
def _gt_mesh(seq_path: str):
    res = compile_text(open(seq_path).read())
    if not res.is_good_solid:
        return None
    return G.mesh_of(res.shape)


def _cd(a, b):
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return 0.5 * (da.mean() + db.mean()), 0.5 * ((da ** 2).mean() + (db ** 2).mean())


def eval_metrics(text: str, gt_seq_path: str) -> dict:
    out = dict(iou128=0.0, iou_best=0.0, cd_l2=float("nan"), cd_l2sq=float("nan"))
    gt = _gt_mesh(gt_seq_path)
    if gt is None:
        return out
    res = compile_text(extract_sequence(text))
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
