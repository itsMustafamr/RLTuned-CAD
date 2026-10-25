"""Stand-in BRep compiler: v3 sequence -> OpenCascade solid.

Our minimal compiler, so the RL pipeline can run before Bhavika's compiler /
reconstruction code arrives. The sequence stores surfaces, 3D curves, vertices and
loops; OCC's ShapeFix tools recover what it leaves out (p-curves, face orientation,
degenerate edges), then the faces are sewn into a shell and a solid.

Conventions verified on chunk 9 (DATA_REPORT.md):
  * wires list signed, 1-based edge references; sign = edge direction in the loop
  * circle/ellipse `sweep`: 0 = arc runs counter-clockwise about axis_dir from v[0]
    to v[1], 1 = clockwise (i.e. counter-clockwise about -axis_dir), 2 = full curve
  * cone semi_angle is in integer degrees

Requires cadquery-ocp 8.x (OCCT 8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int, Array2_double, Array2_gp_Pnt
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.Geom import (
    Geom_BSplineCurve,
    Geom_BSplineSurface,
    Geom_Circle,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Ellipse,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shell, ShapeFix_Solid
from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Solid, TopoDS_Wire

from .seqfmt import Seq, parse

TWO_PI = 2.0 * math.pi


def _static(owner, name):
    """OCP 8 binds class static methods as `name_s` but namespace functions (TopoDS.Shell) without the suffix."""
    return getattr(owner, name + "_s", None) or getattr(owner, name)


_as_shell = _static(TopoDS, "Shell")
_is_closed = _static(BRep_Tool, "IsClosed")
_vol_props = _static(BRepGProp, "VolumeProperties")
_surf_props = _static(BRepGProp, "SurfaceProperties")


@dataclass
class CompileResult:
    stage: str = "start"              # furthest stage reached: edges | faces | shell | ok
    closed: bool = False
    valid: bool = False
    volume: float = 0.0
    area: float = 0.0
    error: str = ""
    shape: object = None              # TopoDS_Solid when stage == "ok"

    @property
    def is_good_solid(self) -> bool:
        return self.stage == "ok" and self.valid and self.closed and self.volume > 1e-6


# ----------------------------------------------------------------------------
def _pnt(p) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def _dir(d) -> gp_Dir:
    return gp_Dir(float(d[0]), float(d[1]), float(d[2]))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def _perp(axis, ref):
    x = ref - float(np.dot(ref, axis)) * axis
    u = _unit(x)
    if u is None:  # ref parallel to axis -> any perpendicular
        trial = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = _unit(trial - float(np.dot(trial, axis)) * axis)
    return u


def _arr1_pnt(pts):
    a = Array1_gp_Pnt(1, len(pts))
    for i, p in enumerate(pts, 1):
        a.SetValue(i, _pnt(p))
    return a


def _arr1_real(vals):
    a = Array1_double(1, len(vals))
    for i, v in enumerate(vals, 1):
        a.SetValue(i, float(v))
    return a


def _arr1_int(vals):
    a = Array1_int(1, len(vals))
    for i, v in enumerate(vals, 1):
        a.SetValue(i, int(v))
    return a


# ----------------------------------------------------------------------------
class _Builder:
    def __init__(self, seq: Seq, tol: float = 2e-3):
        self.seq = seq
        self.tol = tol
        self.b = BRep_Builder()
        self.V = {}
        for i, p in seq.verts.items():
            v = BRepBuilderAPI_MakeVertex(_pnt(p)).Vertex()
            self.b.UpdateVertex(v, tol)
            self.V[i] = v

    # ---------------- curves / edges
    def _edge_from_curve(self, curve, e, p0: float, p1: float):
        V = self.V
        mk = BRepBuilderAPI_MakeEdge(curve, V[e.v0], V[e.v1], p0, p1)
        if mk.IsDone():
            return mk.Edge()
        # fallback: re-project the vertices to get parameters
        try:
            q0 = GeomAPI_ProjectPointOnCurve(_pnt(self.seq.verts[e.v0]), curve).LowerDistanceParameter()
            q1 = GeomAPI_ProjectPointOnCurve(_pnt(self.seq.verts[e.v1]), curve).LowerDistanceParameter()
            mk = BRepBuilderAPI_MakeEdge(curve, V[e.v0], V[e.v1], q0, q1)
            if mk.IsDone():
                return mk.Edge()
        except Exception:
            pass
        return None

    def edge(self, e):
        P = self.seq.verts
        if e.v0 not in P or e.v1 not in P:
            return None
        if e.kind == "line":
            if np.linalg.norm(P[e.v0] - P[e.v1]) < 1e-9:
                return None
            mk = BRepBuilderAPI_MakeEdge(self.V[e.v0], self.V[e.v1])
            return mk.Edge() if mk.IsDone() else None

        if e.kind in ("circle", "ellipse"):
            c = np.asarray(e.p["center"], float)
            a = _unit(np.asarray(e.p["axis_dir"], float))
            if a is None:
                return None
            sweep = int(e.p.get("sweep", 0))
            ax = -a if sweep == 1 else a           # sweep 1 = clockwise about axis_dir
            p0, p1 = P[e.v0], P[e.v1]
            closed = (e.v0 == e.v1) or sweep == 2
            if e.kind == "circle":
                r = float(e.p["r"])
                if r <= 1e-9:
                    return None
                x = _perp(ax, p0 - c)
                y = np.cross(ax, x)
                curve = Geom_Circle(gp_Ax2(_pnt(c), _dir(ax), _dir(x)), r)
                if closed:
                    t1 = TWO_PI
                else:
                    d = p1 - c
                    t1 = math.atan2(float(d @ y), float(d @ x)) % TWO_PI
                    if t1 < 1e-9:
                        t1 = TWO_PI
                return self._edge_from_curve(curve, e, 0.0, t1)
            R, rr = float(e.p["r_major"]), float(e.p["r_minor"])
            X = _perp(ax, np.asarray(e.p["major_dir"], float))
            if rr > R:  # keep OCC's R >= r requirement
                R, rr = rr, R
                X = np.cross(ax, X)
            if rr <= 1e-9:
                return None
            Y = np.cross(ax, X)
            curve = Geom_Ellipse(gp_Ax2(_pnt(c), _dir(ax), _dir(X)), R, rr)

            def tpar(q):
                d = q - c
                return math.atan2(float(d @ Y) / rr, float(d @ X) / R)

            t0 = tpar(p0)
            if closed:
                t1 = t0 + TWO_PI
            else:
                t1 = tpar(p1)
                while t1 <= t0 + 1e-9:
                    t1 += TWO_PI
            return self._edge_from_curve(curve, e, t0, t1)

        if e.kind == "bspline":
            poles = np.asarray(e.p["poles"], float)
            weights = [float(w) for w in e.p.get("weights", [1.0] * len(poles))]
            knots = [float(k) for k in e.p["knots"]]
            mults = [int(m) for m in e.p["mults"]]
            deg = int(e.p["degree"])
            periodic = bool(e.p.get("periodic", False))
            if all(abs(w - 1.0) < 1e-12 for w in weights):
                curve = Geom_BSplineCurve(_arr1_pnt(poles), _arr1_real(knots), _arr1_int(mults), deg, periodic)
            else:
                curve = Geom_BSplineCurve(_arr1_pnt(poles), _arr1_real(weights), _arr1_real(knots), _arr1_int(mults), deg, periodic)
            t0, t1 = [float(t) for t in e.p.get("trim", [knots[0], knots[-1]])]
            return self._edge_from_curve(curve, e, t0, t1)

        return None  # edge_other (degenerate): ShapeFix re-creates these where needed

    # ---------------- surfaces
    def surface(self, f):
        p = f.p
        k = f.kind
        if k == "plane":
            n = _unit(np.asarray(p["n"], float))
            return Geom_Plane(_pnt(n * float(p["d"])), _dir(n))
        if k == "cylinder":
            return Geom_CylindricalSurface(gp_Ax3(_pnt(p["axis_org"]), _dir(p["axis_dir"])), float(p["r"]))
        if k == "cone":
            ang = math.radians(min(max(float(p["semi_angle"]), 0.5), 89.5))
            return Geom_ConicalSurface(gp_Ax3(_pnt(p["apex"]), _dir(p["axis_dir"])), ang, 0.0)
        if k == "sphere":
            return Geom_SphericalSurface(gp_Ax3(_pnt(p["center"]), gp_Dir(0, 0, 1)), float(p["r"]))
        if k == "torus":
            return Geom_ToroidalSurface(gp_Ax3(_pnt(p["center"]), _dir(p["axis_dir"])), float(p["r_major"]), float(p["r_minor"]))
        if k == "bsurf":
            P = np.asarray(p["poles"], float)          # (nu, nv, 3)
            W = np.asarray(p.get("weights", np.ones(P.shape[:2])), float)
            um, vm = [int(x) for x in p["um"]], [int(x) for x in p["vm"]]
            ud, vd = int(p["ud"]), int(p["vd"])
            nu, nv = P.shape[0], P.shape[1]
            uper = (sum(um) - ud - 1) != nu and sum(um[:-1]) == nu
            vper = (sum(vm) - vd - 1) != nv and sum(vm[:-1]) == nv
            poles = Array2_gp_Pnt(1, nu, 1, nv)
            wts = Array2_double(1, nu, 1, nv)
            for i in range(nu):
                for j in range(nv):
                    poles.SetValue(i + 1, j + 1, _pnt(P[i, j]))
                    wts.SetValue(i + 1, j + 1, float(W[i, j]))
            return Geom_BSplineSurface(poles, wts, _arr1_real(p["uk"]), _arr1_real(p["vk"]), _arr1_int(um), _arr1_int(vm), ud, vd, uper, vper)
        raise ValueError(f"unknown surface {k}")


def _volume_area(shape):
    gp = GProp_GProps()
    _vol_props(shape, gp)
    vol = abs(gp.Mass())
    gs = GProp_GProps()
    _surf_props(shape, gs)
    return vol, gs.Mass()


def _first(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    return ex.Current() if ex.More() else None


def finalize(shape, res: CompileResult) -> CompileResult:
    """Validity, closedness and mass properties of the first solid in `shape`.
    Shared by our compiler and the external-compiler hook (compiler.py), so every
    compiler is judged by the same test."""
    sol = _first(shape, TopAbs_SOLID)
    if sol is None:
        res.error = "no solid"
        return res
    shell = _first(sol, TopAbs_SHELL)
    res.closed = bool(shell is not None and _is_closed(_as_shell(shell)))
    res.valid = bool(BRepCheck_Analyzer(sol).IsValid())
    res.volume, res.area = _volume_area(sol)
    res.shape = sol
    res.stage = "ok"
    return res


def build(seq: Seq, tol: float = 2e-3) -> CompileResult:
    res = CompileResult()
    try:
        B = _Builder(seq, tol)
        E: Dict[int, object] = {}
        for i, e in seq.edges.items():
            try:
                E[i] = B.edge(e)
            except Exception:
                E[i] = None
        res.stage = "edges"

        wires_by_face: Dict[int, list] = {}
        for w in seq.wires:
            wires_by_face.setdefault(w.face, []).append(w)

        faces = []
        for fi, f in seq.faces.items():
            ws = sorted(wires_by_face.get(fi, []), key=lambda w: w.r)   # outer loop (r=0) first
            if not ws:
                continue
            try:
                F = TopoDS_Face()
                B.b.MakeFace(F, B.surface(f), tol)
                for w in ws:
                    W = TopoDS_Wire()
                    B.b.MakeWire(W)
                    n_in = 0
                    for s in w.edges:
                        ed = E.get(abs(s) - 1)
                        if ed is None:
                            continue
                        B.b.Add(W, ed if s > 0 else ed.Reversed())
                        n_in += 1
                    if n_in:
                        B.b.Add(F, W)
                sff = ShapeFix_Face(F)          # adds p-curves, fixes loop orientation, splits where needed
                sff.SetPrecision(tol)
                sff.SetMaxTolerance(10 * tol)
                sff.Perform()
                faces.append(sff.Face())
            except Exception:
                continue                        # a face that cannot be built is simply missing; sewing decides
        res.stage = "faces"
        if not faces:
            res.error = "no faces"
            return res

        sew = BRepBuilderAPI_Sewing(tol)
        for F in faces:
            sew.Add(F)
        sew.Perform()
        shell = _first(sew.SewedShape(), TopAbs_SHELL)
        if shell is None:
            res.error = "sewing produced no shell"
            return res
        sfs = ShapeFix_Shell(_as_shell(shell))
        sfs.Perform()
        res.stage = "shell"

        solid = TopoDS_Solid()
        B.b.MakeSolid(solid)
        B.b.Add(solid, _as_shell(sfs.Shell()))
        sfsol = ShapeFix_Solid(solid)           # orients the shell so the volume is positive
        sfsol.Perform()
        return finalize(sfsol.Shape(), res)
    except Exception as ex:
        res.error = f"{type(ex).__name__}: {ex}"[:200]
    return res


def compile_sequence(text: str, tol: float = 2e-3) -> CompileResult:
    return build(parse(text), tol=tol)
