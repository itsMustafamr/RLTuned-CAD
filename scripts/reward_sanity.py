"""Reward sanity test: the GT must score ~1 and corruptions must score lower, roughly
in order of severity. Run before every RL run that changes the reward.

  python scripts/reward_sanity.py --manifest data/processed/manifest.jsonl --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brepgrpo.reward import score  # noqa: E402
from brepgrpo.seqfmt import strip_to_body  # noqa: E402


def jitter(body: str, k: int, rng: random.Random) -> str:
    """Move every vertex by up to +-k quantization steps (geometry drifts, topology intact)."""
    def f(m):
        vals = [int(x) for x in m.group(2).split(",")]
        vals = [min(4094, max(0, v + rng.randint(-k, k))) for v in vals]
        return f"v({m.group(1)},{','.join(map(str, vals))})"
    return re.sub(r"^v\((\d+),([-\d,]+)\)", f, body, flags=re.M)


def drop_face(body: str, rng: random.Random) -> str:
    faces = re.findall(r"^f(\d+) = ", body, flags=re.M)
    fi = rng.choice(faces)
    body = re.sub(rf"^f{fi} = .*\n", "", body, flags=re.M)
    return re.sub(rf"^wire\(\d+, face={fi}, .*\n", "", body, flags=re.M)


def swap_wire_edges(body: str, rng: random.Random) -> str:
    lines = body.splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith("wire(") and l.count(",") >= 5]
    for i in rng.sample(idx, min(3, len(idx))):
        m = re.match(r"^(wire\(.*edges=\[)([^\]]*)(\]\))$", lines[i])
        es = m.group(2).split(", ")
        rng.shuffle(es)
        lines[i] = m.group(1) + ", ".join(es) + m.group(3)
    return "\n".join(lines) + "\n"


def truncate(body: str) -> str:
    return body[: len(body) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest)]
    rows = [r for r in rows if r.get("gt_ok")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]
    box = next((r for r in rows if r["n_faces"] == 6), rows[0])

    res = defaultdict(list)
    t0 = time.time()
    n_calls = 0
    for r in rows:
        body = strip_to_body(open(r["seq_path"]).read())
        gt = r["gt_cache"]
        other = rng.choice([x for x in rows if x["id"] != r["id"]])
        variants = {
            "gt": body,
            "jitter_2": jitter(body, 2, rng),
            "jitter_8": jitter(body, 8, rng),
            "jitter_32": jitter(body, 32, rng),
            "jitter_128": jitter(body, 128, rng),
            "other_model": strip_to_body(open(other["seq_path"]).read()),
            "simple_box": strip_to_body(open(box["seq_path"]).read()) if box["id"] != r["id"] else None,
            "drop_face": drop_face(body, rng),
            "shuffled_wires": swap_wire_edges(body, rng),
            "truncated": truncate(body),
            "empty": "",
        }
        for k, v in variants.items():
            if v is None:
                continue
            out = score(v, gt)
            n_calls += 1
            res[k].append((out["score"], out["valid_solid"], out["iou"], out["s_topo"]))
    print(f"{'variant':16s} {'n':>3s} {'reward':>8s} {'+-':>6s} {'valid':>6s} {'iou':>6s} {'topo':>6s}")
    for k, v in res.items():
        a = np.array(v)
        print(f"{k:16s} {len(a):3d} {a[:,0].mean():8.3f} {a[:,0].std():6.3f} {a[:,1].mean():6.2f} {a[:,2].mean():6.3f} {a[:,3].mean():6.3f}")
    print(f"\n{n_calls} reward calls, {1000 * (time.time() - t0) / n_calls:.0f} ms/call on one core")


if __name__ == "__main__":
    main()
