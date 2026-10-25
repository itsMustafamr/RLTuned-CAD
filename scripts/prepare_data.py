"""Build everything the training needs from the raw chunks Bhavika shares.

Input  (any number of chunks):
  --seq-root   dir(s) containing  <...>/{simple,complex}/<chunk>/<id>/<id>.seq.txt
  --ann-root   dir(s) containing  <chunk>/<id>/<id>_{construction_steps,global_shape,topology_relations}.json

Output (--out, default data/processed):
  manifest.jsonl        one row per model: paths, stats, split, tier, gt_ok
  gt_cache/<id>.npz     reward assets (GT rebuilt by brepgrpo.compiler_occ)
  sft_{train,val}.parquet           verl SFT format (messages)
  rl_train_{8k,16k}.parquet, rl_val.parquet   verl RL format, train sorted by curriculum tier
  test.jsonl            held-out prompts for scripts/eval.py
  stats.json

Rules
  * exact-duplicate sequences are collapsed (one representative per shape) and a
    duplicate group never straddles splits
  * models whose GT does not rebuild into a valid solid are kept for SFT only
  * split: 90 / 3 / 7 (train / val / test), stratified by simple/complex
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from brepgrpo.prompts import SYSTEM_PROMPT, rl_user_prompt, sft_user_prompts  # noqa: E402
from brepgrpo.seqfmt import parse, strip_to_body  # noqa: E402

ANN_KINDS = ("construction_steps", "global_shape", "topology_relations")
_PRETOK = None


def est_tokens(text: str) -> int:
    """Token count without the tokenizer files: the Qwen2-family pre-tokenizer split (every digit is its
    own token) is a lower bound on the real count, the 5% margin keeps GT sequences inside the SFT/RL
    length limits. tests/preflight_nova.py checks this estimate against the real tokenizer on Nova."""
    global _PRETOK
    if _PRETOK is None:
        import regex
        _PRETOK = regex.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
    return int(len(_PRETOK.findall(text)) * 1.05)


def clean_desc(d: str) -> str:
    d = re.sub(r"</?[A-Z_]+_DESC>", "", d)
    return d.strip()


def index_annotations(ann_roots):
    """model id -> folder holding its <id>_<kind>.json files (one directory walk, instead of a glob per model)."""
    index = {}
    for root in ann_roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                for k in ANN_KINDS:
                    if f.endswith(f"_{k}.json"):
                        index.setdefault(f[: -len(f"_{k}.json")], dirpath)
    return index


def load_annotations(ann_dir, mid):
    out = {}
    for k in ANN_KINDS:
        p = os.path.join(ann_dir, f"{mid}_{k}.json")
        if os.path.exists(p):
            try:
                out[k] = clean_desc(json.load(open(p))["description"])
            except (OSError, ValueError, KeyError):
                pass
    return out


def tier_of(n_faces: int, hist: dict) -> int:
    kinds = {k for k, v in hist.items() if v}
    if n_faces >= 30:
        return 3                        # complex
    if kinds <= {"plane"} and n_faces <= 12:
        return 0                        # small planar
    if kinds <= {"plane", "cylinder"}:
        return 1                        # planes + cylinders
    return 2                            # cones / tori / spheres / B-splines


def _gt_job(args):
    mid, seq_path, out_path = args
    try:
        from brepgrpo.gtcache import build_gt_assets
        a = build_gt_assets(open(seq_path).read())
        if a is None:
            return mid, False
        np.savez_compressed(out_path, **a)
        return mid, True
    except Exception:
        return mid, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-root", nargs="+", required=True)
    ap.add_argument("--ann-root", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "processed"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=15000, help="drop GT sequences longer than this (SFT/RL); SFT max_length is 16384")
    ap.add_argument("--skip-gt", action="store_true", help="reuse an existing gt_cache")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "gt_cache"), exist_ok=True)
    files = []
    for root in args.seq_root:
        files += glob.glob(os.path.join(root, "**", "*.seq.txt"), recursive=True)
    files = sorted(set(files))
    print(f"found {len(files)} sequences")

    # ---- per-model rows + exact-duplicate collapsing
    rows, by_hash = {}, defaultdict(list)
    for f in files:
        mid = os.path.basename(f).split(".")[0]
        cat = "complex" if f"{os.sep}complex{os.sep}" in f else "simple"
        body = strip_to_body(open(f).read())
        seq = parse(body)
        hist = Counter(fc.kind for fc in seq.faces.values())
        rows[mid] = dict(id=mid, category=cat, seq_path=os.path.abspath(f), n_faces=len(seq.faces),
                         n_edges=len(seq.edges), n_verts=len(seq.verts), surf_hist=dict(hist),
                         tokens=est_tokens(body), tier=tier_of(len(seq.faces), hist))
        by_hash[hashlib.md5(body.encode()).hexdigest()].append(mid)

    ann_index = index_annotations(args.ann_root)
    keep = []
    for h, mids in by_hash.items():
        rep = None
        for m in sorted(mids):
            ann = load_annotations(ann_index[m], m) if m in ann_index else {}
            if ann:
                rep = m
                rows[m]["ann"] = ann
                break
        if rep is None:
            continue
        rows[rep]["dup_group"] = h[:10]
        rows[rep]["n_duplicates"] = len(mids)
        keep.append(rep)
    print(f"unique shapes with annotations: {len(keep)}  (collapsed {len(files) - len(by_hash)} exact duplicates; "
          f"{len(by_hash) - len(keep)} shapes had no annotation)")

    # ---- GT assets (parallel)
    jobs = [(m, rows[m]["seq_path"], os.path.join(args.out, "gt_cache", f"{m}.npz")) for m in keep
            if rows[m]["tokens"] <= args.max_tokens]
    if args.skip_gt:
        ok = {m: os.path.exists(p) for m, _, p in jobs}
    else:
        ok = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (m, good) in enumerate(ex.map(_gt_job, jobs, chunksize=4)):
                ok[m] = good
                if (i + 1) % 500 == 0:
                    print(f"  gt cache {i + 1}/{len(jobs)}")
    for m in keep:
        rows[m]["gt_ok"] = bool(ok.get(m))
        rows[m]["gt_cache"] = os.path.abspath(os.path.join(args.out, "gt_cache", f"{m}.npz"))
    print(f"GT rebuilt into valid solids: {sum(ok.values())}/{len(jobs)} attempted "
          f"({len(keep) - len(jobs)} skipped as longer than {args.max_tokens} tokens)")

    # ---- split (stratified by category)
    rng = random.Random(args.seed)
    for cat in ("simple", "complex"):
        mids = sorted(m for m in keep if rows[m]["category"] == cat)
        rng.shuffle(mids)
        n = len(mids)
        n_test, n_val = round(0.07 * n), round(0.03 * n)
        for i, m in enumerate(mids):
            rows[m]["split"] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    # ---- write manifest
    with open(os.path.join(args.out, "manifest.jsonl"), "w") as fo:
        for m in keep:
            fo.write(json.dumps(rows[m]) + "\n")

    import pandas as pd

    def body_of(m):
        return strip_to_body(open(rows[m]["seq_path"]).read())

    # ---- SFT parquet (4 prompt variants per shape)
    for split in ("train", "val"):
        recs = []
        for m in keep:
            r = rows[m]
            if r["split"] != split or r["tokens"] > args.max_tokens:
                continue
            target = body_of(m)
            for k, user in sft_user_prompts(r["ann"]):
                recs.append(dict(messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": user},
                                           {"role": "assistant", "content": target}],
                                 id=m, variant=k))
        rng.shuffle(recs)
        pd.DataFrame(recs).to_parquet(os.path.join(args.out, f"sft_{split}.parquet"))
        print(f"sft_{split}: {len(recs)} samples")

    # ---- RL parquet (rich prompt, GT must rebuild), curriculum order for train
    def rl_rec(m, idx):
        r = rows[m]
        return dict(
            data_source="brepvlm_text2seq",
            prompt=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": rl_user_prompt(r["ann"])}],
            ability="cad",
            reward_model=dict(style="rule", ground_truth=json.dumps(dict(id=m, cache=r["gt_cache"]))),
            extra_info=dict(index=idx, id=m, tier=r["tier"], n_faces=r["n_faces"], gt_tokens=r["tokens"],
                            split=r["split"]),
        )

    for budget_name, budget in (("8k", 7300), ("16k", 15000)):
        train = [m for m in keep if rows[m]["split"] == "train" and rows[m]["gt_ok"] and rows[m]["tokens"] <= budget]
        rng.shuffle(train)
        train.sort(key=lambda m: rows[m]["tier"])         # stable: random within tier, tiers in order
        pd.DataFrame([rl_rec(m, i) for i, m in enumerate(train)]).to_parquet(
            os.path.join(args.out, f"rl_train_{budget_name}.parquet"))
        print(f"rl_train_{budget_name}: {len(train)} prompts  tiers={dict(Counter(rows[m]['tier'] for m in train))}")
    val = [m for m in keep if rows[m]["split"] == "val" and rows[m]["gt_ok"] and rows[m]["tokens"] <= 7300]
    pd.DataFrame([rl_rec(m, i) for i, m in enumerate(val)]).to_parquet(os.path.join(args.out, "rl_val.parquet"))

    # ---- test set for eval.py (all test shapes with a rebuildable GT)
    with open(os.path.join(args.out, "test.jsonl"), "w") as fo:
        for m in keep:
            r = rows[m]
            if r["split"] == "test" and r["gt_ok"]:
                fo.write(json.dumps(dict(id=m, category=r["category"], tier=r["tier"], tokens=r["tokens"],
                                         n_faces=r["n_faces"], gt_cache=r["gt_cache"], seq_path=r["seq_path"],
                                         messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                                   {"role": "user", "content": rl_user_prompt(r["ann"])}])) + "\n")

    stats = dict(
        sequences=len(files), unique_shapes=len(by_hash), with_annotations=len(keep), gt_ok=int(sum(ok.values())),
        split=dict(Counter(rows[m]["split"] for m in keep)),
        tiers=dict(Counter(rows[m]["tier"] for m in keep)),
        category=dict(Counter(rows[m]["category"] for m in keep)),
    )
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
