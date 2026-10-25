"""Unit tests that need only the repo + OCP (no GPU, no verl):  python -m pytest -q tests/test_core.py
Uses the sequences in data/processed/manifest.jsonl (run scripts/prepare_data.py first)."""
import json, os, sys
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from brepgrpo import checks, seqfmt  # noqa: E402
from brepgrpo.compiler import compile_text  # noqa: E402
from brepgrpo.reward import score  # noqa: E402

MAN = os.path.join(os.environ.get("DATA", os.path.join(REPO, "data", "processed")), "manifest.jsonl")
ROWS = [json.loads(l) for l in open(MAN)][:60] if os.path.exists(MAN) else []
pytestmark = pytest.mark.skipif(not ROWS, reason="run scripts/prepare_data.py first")


def test_parse_roundtrip_counts():
    for r in ROWS[:30]:
        s = seqfmt.parse(open(r["seq_path"]).read())
        assert s.complete and s.n_bad_lines == 0
        assert len(s.faces) == r["n_faces"]


def test_gt_topology_is_perfect():
    for r in ROWS[:30]:
        t = checks.topology_scores(seqfmt.parse(open(r["seq_path"]).read()))
        assert t["s_topo"] == pytest.approx(1.0)


def test_gt_compiles_and_scores_one():
    good = [r for r in ROWS if r["gt_ok"]][:10]
    for r in good:
        assert compile_text(open(r["seq_path"]).read()).is_good_solid
        body = seqfmt.strip_to_body(open(r["seq_path"]).read())
        assert score(body, r["gt_cache"])["score"] > 0.99


def test_garbage_scores_zero():
    r = [r for r in ROWS if r["gt_ok"]][0]
    assert score("hello", r["gt_cache"])["score"] == 0.0
    assert score("model(3)\nv(0,1,2,3)\n", r["gt_cache"])["score"] == 0.0   # no end_model()
