"""Compiler dispatch: our OCC stand-in by default, Bhavika's compiler when it arrives.

Point BREP_COMPILER_MODULE at an importable module (or a .py path) that exposes

    compile_sequence(text: str) -> OCP TopoDS_Shape | None      # the rebuilt solid, or None

`text` is the sequence body (no comment lines, no quantization(...) header), exactly
what the policy writes. Validity, closedness and volume are then judged by the same
`finalize` for every compiler, so rewards stay comparable. Rebuild the GT cache
(scripts/prepare_data.py) after switching compilers.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from functools import lru_cache

from .compiler_occ import CompileResult, finalize
from .compiler_occ import compile_sequence as _compile_occ


@lru_cache(maxsize=1)
def _external():
    name = os.environ.get("BREP_COMPILER_MODULE", "").strip()
    if not name:
        return None
    if name.endswith(".py"):
        spec = importlib.util.spec_from_file_location("brep_external_compiler", name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(name)
    return mod.compile_sequence


def compile_text(text: str) -> CompileResult:
    ext = _external()
    if ext is None:
        return _compile_occ(text)
    res = CompileResult()
    try:
        shape = ext(text)
        if shape is None:
            res.error = "external compiler returned None"
            return res
        return finalize(shape, res)
    except Exception as ex:
        res.error = f"{type(ex).__name__}: {ex}"[:200]
    return res
