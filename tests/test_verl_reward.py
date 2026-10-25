"""Smoke test for the verl reward entry point (isolated child + fallbacks).
   python tests/test_verl_reward.py --manifest data/processed/manifest.jsonl"""
import argparse, importlib.util, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from brepgrpo.seqfmt import strip_to_body  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("vr", os.path.join(REPO, "brepgrpo", "verl_reward.py"))
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r["gt_ok"]][:5]
    t = time.time()
    for r in rows:
        gt = json.dumps(dict(id=r["id"], cache=r["gt_cache"]))
        out = vr.compute_score("brepvlm_text2seq", strip_to_body(open(r["seq_path"]).read()), gt, {})
        assert out["score"] > 0.99 and out["timeout"] == 0.0, (r["id"], out)
    print(f"GT through isolated child: OK ({time.time() - t:.2f}s for {len(rows)} incl. child start)")
    out = vr.compute_score("x", "garbage", json.dumps(dict(id="x", cache="/nope.npz")), {})
    assert out["score"] == 0.0
    out = vr.compute_score("x", strip_to_body(open(rows[0]["seq_path"]).read()), json.dumps(dict(id="x", cache="/nope.npz")), {})
    assert out["timeout"] == 1.0 and 0.0 < out["score"] <= 0.4
    print("fallbacks: OK   keys:", sorted(out))


if __name__ == "__main__":
    main()
