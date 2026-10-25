"""Parse / dequantize BRepVLM v3 sequences (12-bit, protocol version 3).

A sequence looks like:

    # CAD model sequence (code format)
    quantization(version=3, enabled=True, ..., bits=12, coord=[-1.0, 1.0], ...)
    model(3)
    solid(0, shells=[0])
    shell(0, faces=[0, 1, 2])
    v(0,1303,2540,0)
    f0 = plane(0, n=[2047, 4094, 2047], d=2540, centroid=[2047, 2540, 2047])
    wire(0, face=0, r=0, edges=[-1, -2, 3, 4])      # signed, 1-based edge refs
    e0 = line(0, v=[0,1])
    end_model()

The model is trained on the BODY only (everything after the quantization line), and
that body is what the compiler and the reward receive (`strip_to_body`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

# protocol constants (from the chunk's norm.json / seq header)
QMAX = 4094.0                      # ordinary_max_index
GRID_COORD = (-1.0, 1.0)
GRID_RADIUS = (0.0, 2.0)
# angles (cone semi_angle) are stored as integer degrees 0..180

COORD_KEYS = {"centroid", "center", "axis_org", "apex", "poles"}
DIR_KEYS = {"n", "axis_dir", "major_dir"}
RADIUS_KEYS = {"r", "r_major", "r_minor"}
SURFACE_KINDS = ("plane", "cylinder", "cone", "sphere", "torus", "bsurf")   # ordered: checks.py builds histograms in this order
CURVE_KINDS = ("line", "circle", "ellipse", "bspline", "edge_other")


def dq_coord(x):
    lo, hi = GRID_COORD
    return lo + (hi - lo) * np.asarray(x, dtype=float) / QMAX


def dq_radius(x):
    lo, hi = GRID_RADIUS
    return lo + (hi - lo) * np.asarray(x, dtype=float) / QMAX


def dq_dir(x):
    v = dq_coord(x)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# --------------------------------------------------------------------------
# tiny argument parser:  name(arg, arg, key=value, key=[...], ...)
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)|(?P<id>[A-Za-z_]\w*)|(?P<p>[\[\](),=]))")


class _ArgParser:
    def __init__(self, s: str):
        self.toks: List[Tuple[str, str]] = []
        pos = 0
        s = s.strip()
        while pos < len(s):
            m = _TOKEN.match(s, pos)
            if not m or m.end() == pos:
                raise ValueError(f"bad token at {pos}: {s[pos:pos + 20]!r}")
            kind = m.lastgroup
            self.toks.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, val=None):
        k, v = self.peek()
        if k is None or (val is not None and v != val):
            raise ValueError(f"expected {val!r}, got {v!r}")
        self.i += 1
        return k, v

    def value(self):
        k, v = self.peek()
        if v == "[":
            self.take("[")
            out = []
            while self.peek()[1] != "]":
                out.append(self.value())
                if self.peek()[1] == ",":
                    self.take(",")
            self.take("]")
            return out
        if k == "num":
            self.take()
            return float(v) if any(c in v for c in ".eE") else int(v)
        if k == "id":
            self.take()
            return {"True": True, "False": False, "None": None}.get(v, v)
        raise ValueError(f"unexpected {v!r}")

    def args(self):
        pos, kw = [], {}
        while self.peek()[0] is not None:
            k, v = self.peek()
            if k == "id" and self.i + 1 < len(self.toks) and self.toks[self.i + 1][1] == "=":
                self.take()
                self.take("=")
                kw[v] = self.value()
            else:
                pos.append(self.value())
            if self.peek()[1] == ",":
                self.take(",")
        return pos, kw


def parse_call(s: str):
    m = re.match(r"^\s*(?:(?P<lhs>[fe]\d+)\s*=\s*)?(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)\s*$", s)
    if not m:
        raise ValueError(f"not a call: {s[:60]!r}")
    pos, kw = _ArgParser(m.group("args")).args() if m.group("args").strip() else ([], {})
    return m.group("lhs"), m.group("name"), pos, kw


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Face:
    idx: int
    kind: str
    p: Dict[str, Any]


@dataclass
class Edge:
    idx: int
    kind: str
    v0: int
    v1: int
    p: Dict[str, Any]


@dataclass
class Wire:
    idx: int
    face: int
    r: int                 # 0 = outer loop, >0 = inner loop
    edges: List[int]       # signed 1-based


@dataclass
class Seq:
    verts: Dict[int, np.ndarray] = field(default_factory=dict)
    faces: Dict[int, Face] = field(default_factory=dict)
    wires: List[Wire] = field(default_factory=list)
    edges: Dict[int, Edge] = field(default_factory=dict)
    shells: Dict[int, List[int]] = field(default_factory=dict)
    solids: Dict[int, List[int]] = field(default_factory=dict)
    complete: bool = False          # saw end_model()
    n_bad_lines: int = 0
    n_lines: int = 0


def _dequantize(kw: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword parameters of a surface or curve, integers -> real values (edge endpoints `v` excluded)."""
    p: Dict[str, Any] = {}
    for k, v in kw.items():
        if k == "v":
            continue
        if k in DIR_KEYS:
            p[k] = dq_dir(v)
        elif k in RADIUS_KEYS:
            p[k] = float(dq_radius(v))
        elif k == "d":                      # plane offset along n
            p[k] = float(dq_coord(v))
        elif k == "semi_angle":             # cone, whole degrees
            p[k] = float(v)
        elif k in COORD_KEYS:
            p[k] = dq_coord(v)
        else:                               # bspline knots / weights / mults / degree: stored as-is
            p[k] = v
    return p


def strip_to_body(text: str) -> str:
    """Remove comment lines and the quantization(...) header; keep the rest verbatim."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("quantization("):
            continue
        out.append(line.rstrip())
    body = "\n".join(out).strip("\n")
    return body + "\n"


def extract_sequence(response: str) -> str:
    """Pull the sequence out of a model response (tolerates ``` fences / chatter)."""
    t = response
    m = re.search(r"```(?:[a-zA-Z]*)\n(.*?)(?:```|$)", t, flags=re.S)
    if m:
        t = m.group(1)
    start = t.find("model(")
    if start > 0:
        t = t[start:]
    end = t.find("end_model()")
    if end >= 0:
        t = t[: end + len("end_model()")]
    return t


def parse(text: str) -> Seq:
    """Lenient parser: bad lines are counted, not fatal."""
    seq = Seq()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        seq.n_lines += 1
        try:
            if line.startswith("v("):
                _, _, pos, _ = parse_call(line)
                if len(pos) != 4:
                    raise ValueError("vertex arity")
                seq.verts[int(pos[0])] = dq_coord(pos[1:4])
                continue
            lhs, name, pos, kw = parse_call(line)
            if name == "end_model":
                seq.complete = True
            elif name in ("model", "quantization"):
                pass
            elif name == "solid":
                seq.solids[int(pos[0])] = list(kw.get("shells", []))
            elif name == "shell":
                seq.shells[int(pos[0])] = list(kw.get("faces", []))
            elif name == "wire":
                seq.wires.append(Wire(int(pos[0]), int(kw["face"]), int(kw.get("r", 0)), [int(x) for x in kw["edges"]]))
            elif name in SURFACE_KINDS:
                idx = int(pos[0])
                seq.faces[idx] = Face(idx, name, _dequantize(kw))
            elif name in CURVE_KINDS:
                idx = int(pos[0])
                v = kw["v"]
                seq.edges[idx] = Edge(idx, name, int(v[0]), int(v[1]), _dequantize(kw))
            else:
                raise ValueError(f"unknown statement {name}")
        except Exception:
            seq.n_bad_lines += 1
    return seq
