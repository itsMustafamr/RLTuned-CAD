"""Side-by-side table of eval runs:   python scripts/compare.py sft grpo_8k [more names]"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [("invalid_ratio", "IR (lower)"), ("iou_mean", "IoU"), ("iou_best_mean", "IoU_best"), ("cd_median_x1e3", "CD-Faces med x1e3"),
        ("cd_mean_x1e3", "CD-Faces mean x1e3"), ("cd_vert_median_x1e3", "CD-Vert med x1e3"), ("cd_edge_median_x1e3", "CD-Edges med x1e3"),
        ("reward_mean", "reward")]

names = sys.argv[1:]
lines = []
for subset in ("all", "category=simple", "category=complex"):
    lines.append(f"\n**{subset}**\n")
    lines.append("| run | n | " + " | ".join(c[1] for c in COLS) + " |")
    lines.append("|---|---|" + "---|" * len(COLS))
    for n in names:
        p = os.path.join(REPO, "results", n, "summary.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)).get(subset)
        if not d:
            continue
        vals = ["-" if d.get(k) is None else f"{d[k]:.3f}" for k, _ in COLS]
        lines.append(f"| {n} | {d['n']} | " + " | ".join(vals) + " |")
out = "\n".join(lines)
print(out)
open(os.path.join(REPO, "results", "comparison.md"), "w").write(out + "\n")
