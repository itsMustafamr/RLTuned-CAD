"""Geometry utilities: tessellation, surface/edge sampling, Chamfer, voxel IoU.

All comparisons happen in the sequence's canonical frame (bbox-centred, longest
side = 2, so coordinates live in [-1, 1]). No alignment is applied for the
training reward, on purpose: an alignment-invariant reward would let the policy
ignore position, scale and orientation. Evaluation (metrics_eval.py) additionally
reports the best IoU over the 24 axis-aligned rotations, as ReCAD does.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location

from .compiler_occ import TopoDS, _Builder, _static
from .seqfmt import Seq

_as_face = _static(TopoDS, "Face")
_triangulation = _static(BRep_Tool, "Triangulation")


# ----------------------------------------------------------------------------- mesh
def mesh_of(shape, lin_defl: float = 0.004, ang_defl: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, False)
    verts, tris, off = [], [], 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = _as_face(ex.Current())
        loc = TopLoc_Location()
        tri = _triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n = tri.NbNodes()
            pts = np.empty((n, 3))
            for i in range(1, n + 1):
                p = tri.Node(i).Transformed(trsf)
                pts[i - 1] = (p.X(), p.Y(), p.Z())
            m = tri.NbTriangles()
            t = np.empty((m, 3), dtype=np.int64)
            for i in range(1, m + 1):
                a, b, c = tri.Triangle(i).Get()
                t[i - 1] = (a - 1 + off, b - 1 + off, c - 1 + off)
            verts.append(pts)
            tris.append(t)
            off += n
        ex.Next()
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(verts), np.concatenate(tris)


def sample_surface(V: np.ndarray, T: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(T) == 0:
        return np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = area.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.zeros((0, 3), np.float32)
    idx = rng.choice(len(T), size=n, p=area / tot)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    return pts.astype(np.float32)


# ----------------------------------------------------------------------------- edges
def sample_edges(seq: Seq, n_total: int = 1500, tol: float = 2e-3) -> np.ndarray:
    """Points along every edge that can be built from the sequence (no faces needed)."""
    B = _Builder(seq, tol)
    curves = []
    for e in seq.edges.values():
        try:
            ed = B.edge(e)
        except Exception:
            ed = None
        if ed is None:
            continue
        try:
            ad = BRepAdaptor_Curve(ed)
            t0, t1 = ad.FirstParameter(), ad.LastParameter()
            ts = np.linspace(t0, t1, 9)
            pts = np.array([[ad.Value(t).X(), ad.Value(t).Y(), ad.Value(t).Z()] for t in ts])
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            curves.append((ad, t0, t1, max(length, 1e-6)))
        except Exception:
            continue
    if not curves:
        return np.zeros((0, 3), np.float32)
    total = sum(c[3] for c in curves)
    out = []
    for ad, t0, t1, L in curves:
        k = max(2, int(round(n_total * L / total)))
        for t in np.linspace(t0, t1, k):
            p = ad.Value(t)
            out.append((p.X(), p.Y(), p.Z()))
    return np.asarray(out, np.float32)


# ----------------------------------------------------------------------------- metrics
def chamfer(a: np.ndarray, b: np.ndarray, n_query: Optional[int] = None, seed: int = 0) -> float:
    """Symmetric mean nearest-neighbour distance (not squared)."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    rng = np.random.default_rng(seed)
    qa = a if n_query is None or len(a) <= n_query else a[rng.choice(len(a), n_query, replace=False)]
    qb = b if n_query is None or len(b) <= n_query else b[rng.choice(len(b), n_query, replace=False)]
    d_ab, _ = cKDTree(b).query(qa, k=1)
    d_ba, _ = cKDTree(a).query(qb, k=1)
    return 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))


def occupancy_from_mesh(V: np.ndarray, T: np.ndarray, res: int = 64, bound: float = 1.05) -> np.ndarray:
    """Exact solid occupancy of a closed triangle mesh: cast one ray per (x, y) voxel column along +z
    and fill between alternate crossings (even-odd rule). Voxel centres are tested, so IoU(A, A) == 1."""
    occ = np.zeros((res, res, res), dtype=bool)
    if len(T) == 0:
        return occ
    h = 2.0 * bound / res
    centers = -bound + (np.arange(res) + 0.5) * h
    # tiny irrational offsets so rays never pass exactly through mesh edges/vertices
    ox, oy = 1.234567e-6, 2.345678e-6
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    d = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    keep = np.abs(d) > 1e-14
    A, B, C, d = A[keep], B[keep], C[keep], d[keep]
    if len(A) == 0:
        return occ
    xmin = np.minimum(np.minimum(A[:, 0], B[:, 0]), C[:, 0]); xmax = np.maximum(np.maximum(A[:, 0], B[:, 0]), C[:, 0])
    ymin = np.minimum(np.minimum(A[:, 1], B[:, 1]), C[:, 1]); ymax = np.maximum(np.maximum(A[:, 1], B[:, 1]), C[:, 1])
    i0 = np.clip(np.ceil((xmin - ox + bound) / h - 0.5), 0, res).astype(np.int64)
    i1 = np.clip(np.floor((xmax - ox + bound) / h - 0.5), -1, res - 1).astype(np.int64)
    j0 = np.clip(np.ceil((ymin - oy + bound) / h - 0.5), 0, res).astype(np.int64)
    j1 = np.clip(np.floor((ymax - oy + bound) / h - 0.5), -1, res - 1).astype(np.int64)
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    cnt = ni * nj
    if cnt.sum() == 0:
        return occ
    tri = np.repeat(np.arange(len(A)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ii = i0[tri] + k % np.maximum(ni[tri], 1)
    jj = j0[tri] + k // np.maximum(ni[tri], 1)
    px, py = centers[ii] + ox, centers[jj] + oy
    a, b, c, dd = A[tri], B[tri], C[tri], d[tri]
    w1 = ((px - a[:, 0]) * (c[:, 1] - a[:, 1]) - (py - a[:, 1]) * (c[:, 0] - a[:, 0])) / dd
    w2 = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (b[:, 1] - a[:, 1]) * (px - a[:, 0])) / dd
    inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
    if not inside.any():
        return occ
    z = (a[:, 2] + w1 * (b[:, 2] - a[:, 2]) + w2 * (c[:, 2] - a[:, 2]))[inside]
    ray = (ii * res + jj)[inside]
    order = np.lexsort((z, ray))
    ray, z = ray[order], z[order]
    starts = np.flatnonzero(np.r_[True, ray[1:] != ray[:-1]])
    ends = np.r_[starts[1:], len(ray)]
    for s, e in zip(starts, ends):
        zs = z[s:e]
        if len(zs) < 2:
            continue
        r = ray[s]
        col = occ[r // res, r % res]
        for q in range(0, len(zs) - 1, 2):
            k0 = int(np.ceil((zs[q] + bound) / h - 0.5))
            k1 = int(np.floor((zs[q + 1] + bound) / h - 0.5))
            if k1 >= k0:
                col[max(k0, 0): min(k1, res - 1) + 1] = True
    return occ


def solid_occupancy(V: np.ndarray, T: np.ndarray, res: int = 64, bound: float = 1.05, n_surf: int = 30000) -> np.ndarray:
    """Inside voxels (even-odd fill) OR voxels touched by the surface. The surface term keeps
    parts thinner than a voxel from vanishing; the fixed seed makes identical meshes give
    identical grids (IoU(A, A) == 1)."""
    occ = occupancy_from_mesh(V, T, res, bound)
    pts = sample_surface(V, T, n_surf, seed=0)
    if len(pts):
        idx = np.floor((pts + bound) / (2.0 * bound) * res).astype(np.int64)
        m = np.all((idx >= 0) & (idx < res), axis=1)
        idx = idx[m]
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else 0.0


def cube_rotations():
    mats = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats  # 24 proper rotations


def soft_exp(x: float, floor: float, scale: float) -> float:
    """exp(-(x - floor)/scale) clipped at 1: distances up to `floor` count as perfect."""
    if not math.isfinite(x):
        return 0.0
    return float(math.exp(-max(0.0, x - floor) / scale))
