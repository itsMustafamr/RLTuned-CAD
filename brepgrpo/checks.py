"""Cheap, OCC-free checks on a parsed sequence.

Every ground-truth sequence in chunk 9 passes all of these (DATA_REPORT.md), so
they are necessary conditions we can reward before any geometry is built.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np

from .seqfmt import SURFACE_KINDS, Seq


def topology_scores(seq: Seq) -> Dict[str, float]:
    F, E, V = seq.faces, seq.edges, seq.verts
    n_ref, n_bad = 0, 0
    uses: Counter = Counter()
    for w in seq.wires:
        n_ref += 1
        n_bad += w.face not in F
        for s in w.edges:
            n_ref += 1
            if s == 0 or (abs(s) - 1) not in E:
                n_bad += 1
            else:
                uses[abs(s) - 1] += 1
    for e in E.values():
        n_ref += 2
        n_bad += (e.v0 not in V) + (e.v1 not in V)
    for faces in seq.shells.values():
        n_ref += len(faces)
        n_bad += sum(f not in F for f in faces)
    refs_ok = 1.0 - n_bad / max(1, n_ref)

    # every wire must close when its oriented edges are chained vertex to vertex
    n_closed = 0
    for w in seq.wires:
        segs, ok = [], True
        for s in w.edges:
            e = E.get(abs(s) - 1)
            if e is None:
                ok = False
                break
            segs.append((e.v0, e.v1) if s > 0 else (e.v1, e.v0))
        if ok and segs:
            ok = all(segs[i][1] == segs[(i + 1) % len(segs)][0] for i in range(len(segs)))
        n_closed += bool(ok and segs)
    closed_frac = n_closed / len(seq.wires) if seq.wires else 0.0

    # closed 2-manifold: every non-degenerate edge appears in exactly two wire slots
    nondeg = [i for i, e in E.items() if e.kind != "edge_other"]
    manifold_frac = (sum(uses.get(i, 0) == 2 for i in nondeg) / len(nondeg)) if nondeg else 0.0

    # every face has exactly one outer loop (r == 0)
    outer = Counter(w.face for w in seq.wires if w.r == 0)
    outer_frac = (sum(outer.get(f, 0) == 1 for f in F) / len(F)) if F else 0.0

    s = (closed_frac + manifold_frac + outer_frac) / 3.0
    return dict(
        refs_ok=float(refs_ok),
        wires_closed=closed_frac,
        manifold=manifold_frac,
        outer_loops=outer_frac,
        s_topo=float(refs_ok) * s,
    )


def structure_signature(seq: Seq) -> Dict[str, object]:
    hist = Counter(f.kind for f in seq.faces.values())
    return dict(
        F=len(seq.faces),
        E=sum(1 for e in seq.edges.values() if e.kind != "edge_other"),
        V=len(seq.verts),
        hist=[int(hist.get(t, 0)) for t in SURFACE_KINDS],
    )


def structure_similarity(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    """1.0 when face/edge/vertex counts and the surface-type mix match the target."""
    def count_sim(a, b):
        return float(np.exp(-abs(a - b) / max(1.0, float(b))))

    cs = (count_sim(pred["F"], gt["F"]) + count_sim(pred["E"], gt["E"]) + count_sim(pred["V"], gt["V"])) / 3.0
    hp = np.asarray(pred["hist"], float)
    hg = np.asarray(gt["hist"], float)
    if hp.sum() == 0 or hg.sum() == 0:
        ts = 0.0
    else:
        ts = 1.0 - 0.5 * float(np.abs(hp / hp.sum() - hg / hg.sum()).sum())
    return 0.5 * cs + 0.5 * ts
