"""Precompute per-model ground-truth assets used by the reward (one .npz per model).

The GT geometry is the GT *sequence* rebuilt by the same compiler the policy is
scored with, so a perfect output can reach reward ~1.0 and quantization/compiler
quirks cancel out. (Evaluation against the original STEP is done separately.)
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np

from . import geometry as G
from .checks import structure_signature
from .compiler import compile_text
from .seqfmt import parse

N_SURF = 20000
N_EDGE = 1500
RES = 64


def build_gt_assets(seq_text: str, seed: int = 0) -> Optional[Dict[str, np.ndarray]]:
    seq = parse(seq_text)
    res = compile_text(seq_text)
    if not res.is_good_solid:
        return None
    V, T = G.mesh_of(res.shape)
    s1 = G.sample_surface(V, T, N_SURF, seed=seed)
    s2 = G.sample_surface(V, T, N_SURF, seed=seed + 1)
    w1 = G.sample_edges(seq, N_EDGE)
    if len(s1) == 0 or len(w1) == 0:
        return None
    occ = G.solid_occupancy(V, T, RES)
    sig = structure_signature(seq)
    return dict(
        surf=s1.astype(np.float16),
        # noise floor: Chamfer between two independent samplings of the same GT surface
        cd_floor=np.float32(G.chamfer(s1, s2, n_query=4000, seed=seed)),
        occ=np.packbits(occ.reshape(-1)),
        wire=w1.astype(np.float16),
        sig=np.frombuffer(json.dumps(sig).encode(), dtype=np.uint8),
        volume=np.float32(res.volume),
        area=np.float32(res.area),
    )


def load_gt_assets(path: str) -> Dict[str, object]:
    z = np.load(path)
    return dict(
        surf=z["surf"].astype(np.float32),
        cd_floor=float(z["cd_floor"]),
        occ=np.unpackbits(z["occ"])[: RES ** 3].reshape(RES, RES, RES).astype(bool),
        wire=z["wire"].astype(np.float32),
        sig=json.loads(bytes(z["sig"]).decode()),
        volume=float(z["volume"]),
        area=float(z["area"]),
    )
