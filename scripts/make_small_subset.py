"""Small, short-answer subset of data/processed for the 1-GPU pipeline test (SFT -> eval -> GRPO smoke).

Keeps only tier 0-1 shapes (planar, planes + cylinders) with short ground truth, so one A100 can run every
stage in about an hour. The numbers it produces test the pipeline; they are not results.

    python scripts/make_small_subset.py                 # data/processed -> data/processed_small
    DATA=$PWD/data/processed_small SFT_EXP=sft_small_test ...   (then the launchers, see the README)

Defaults reproduce the 2026-10-29 Nova desktop test: 200 train shapes (800 SFT rows), 20 val shapes
(80 rows), 64 RL prompts, 16 RL val prompts, 32 test shapes.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "data", "processed"))
    ap.add_argument("--dst", default=os.path.join(REPO, "data", "processed_small"))
    ap.add_argument("--sft-shapes", type=int, default=200)
    ap.add_argument("--val-shapes", type=int, default=20)
    ap.add_argument("--rl-prompts", type=int, default=64)
    ap.add_argument("--rl-val", type=int, default=16)
    ap.add_argument("--test", type=int, default=32)
    ap.add_argument("--sft-max-tokens", type=int, default=3000, help="GT length cap for SFT shapes")
    ap.add_argument("--rl-max-tokens", type=int, default=1800, help="GT length cap so an answer fits the 2k smoke budget")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    man = [json.loads(line) for line in open(os.path.join(a.src, "manifest.jsonl"))]
    for name, split, cap in (("sft_train", "train", a.sft_shapes), ("sft_val", "val", a.val_shapes)):
        ids = sorted(r["id"] for r in man if r["split"] == split and r["tier"] <= 1 and r["tokens"] <= a.sft_max_tokens)
        keep = set(ids[:cap])
        df = pd.read_parquet(os.path.join(a.src, f"{name}.parquet"))
        df = df[df["id"].isin(keep)].reset_index(drop=True)
        df.to_parquet(os.path.join(a.dst, f"{name}.parquet"))
        print(f"{name}: {len(df)} rows from {len(keep)} shapes")

    # rl_train is in curriculum order, so the first prompts that pass the filter are tier 0
    rl = pd.read_parquet(os.path.join(a.src, "rl_train_8k.parquet"))
    rl = rl[rl["extra_info"].apply(lambda e: e["tier"] <= 1 and e["gt_tokens"] <= a.rl_max_tokens)]
    rl = rl.head(a.rl_prompts).reset_index(drop=True)
    rl.to_parquet(os.path.join(a.dst, "rl_train_8k.parquet"))
    print(f"rl_train_8k: {len(rl)} prompts")

    val = pd.read_parquet(os.path.join(a.src, "rl_val.parquet"))
    val = val[val["extra_info"].apply(lambda e: e["gt_tokens"] <= a.rl_max_tokens)].head(a.rl_val).reset_index(drop=True)
    val.to_parquet(os.path.join(a.dst, "rl_val.parquet"))
    print(f"rl_val: {len(val)} prompts")

    n = 0
    with open(os.path.join(a.src, "test.jsonl")) as fi, open(os.path.join(a.dst, "test.jsonl"), "w") as fo:
        for line in fi:
            if n < a.test and json.loads(line)["tokens"] <= a.rl_max_tokens:
                fo.write(line)
                n += 1
    print(f"test: {n} shapes")


if __name__ == "__main__":
    main()
