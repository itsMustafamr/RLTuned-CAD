"""Generate with vLLM and score with the same verifiable reward used in RL.

  # held-out test set, greedy (the numbers for the table)
  python scripts/eval.py --model <hf dir> --name sft --test data/processed/test.jsonl
  # pipeline check without a GPU: the "model" answers with the GT sequence, every metric must be ~perfect
  python scripts/eval.py --name oracle --oracle

Sharding (one process per GPU, then merge):
  CUDA_VISIBLE_DEVICES=i python scripts/eval.py ... --shard i --num-shards k
  python scripts/eval.py --name sft --merge --num-shards k
Outputs: results/<name>/{predictions*.jsonl, summary.json, summary.md}
Metrics: invalid_ratio (ReCAD IR), iou_mean (128^3, invalid = 0), iou_best_mean (best of 24 axis rotations),
         CD = mean squared NN distance x 1e3 on 30k points (valid solids only), reward_mean (training reward)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_items(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path)]
    import pandas as pd
    df = pd.read_parquet(path)
    items = []
    for _, r in df.iterrows():
        gt = json.loads(r["reward_model"]["ground_truth"])
        ex = r["extra_info"]
        items.append(dict(id=gt["id"], gt_cache=gt["cache"], tier=int(ex.get("tier", -1)), n_faces=int(ex.get("n_faces", 0)),
                          category="complex" if int(ex.get("n_faces", 0)) >= 30 else "simple",
                          messages=[dict(m) for m in r["prompt"]]))
    return items


def _score(args):
    from brepgrpo.reward import _empty, score
    text, gt_path, seq_path = args
    try:
        out = score(text, gt_path)
    except Exception:
        out = _empty()
    if seq_path:  # paper metrics (denser than the reward); only defined for valid solids
        try:
            from brepgrpo.metrics_eval import eval_metrics
            out.update(eval_metrics(text, seq_path) if out.get("valid_solid") else
                       dict(iou128=0.0, iou_best=0.0, cd_l2=float("nan"), cd_l2sq=float("nan")))
        except Exception:
            pass
    return out


def generate(items, a):
    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_tokens + 1536,
              gpu_memory_utilization=a.gpu_util, seed=a.seed)
    try:  # text-only prompts: skip the vision encoder's memory profiling
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception:
        llm = LLM(**kw)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=1.0 if a.temperature == 0 else a.top_p,
                        max_tokens=a.max_tokens, seed=a.seed)
    outs = llm.chat([it["messages"] for it in items], sp, use_tqdm=True)
    return [[o.text for o in out.outputs] for out in outs]


def score_all(items, texts, workers):
    import multiprocessing as mp

    from pebble import ProcessPool
    jobs = [(t, it["gt_cache"], it.get("seq_path")) for it, ts in zip(items, texts) for t in ts]
    res = []
    with ProcessPool(max_workers=workers, max_tasks=200, context=mp.get_context("spawn")) as pool:
        fut = pool.map(_score, jobs, timeout=120)
        it = fut.result()
        while True:
            try:
                res.append(next(it))
            except StopIteration:
                break
            except Exception:
                from brepgrpo.reward import _empty
                res.append(_empty())
    out, k = [], 0
    for ts in texts:
        out.append(res[k:k + len(ts)])
        k += len(ts)
    return out


def summarize(rows, n):
    def block(rs):
        first = [r["scores"][0] for r in rs]
        valid = np.array([s["valid_solid"] for s in first])
        has_eval = all("iou128" in s for s in first)
        iou = np.array([s["iou128"] if has_eval else s["iou"] for s in first])
        iou_best = np.array([s.get("iou_best", 0.0) for s in first])
        cd = np.array([s["cd_l2sq"] if has_eval else s["cd"] for s, v in zip(first, valid) if v])
        cd = cd[np.isfinite(cd)]
        d = dict(
            n=len(rs),
            invalid_ratio=float(1 - valid.mean()) if len(valid) else None,
            iou_mean=float(iou.mean()) if len(iou) else None,
            iou_mean_valid=float(iou[valid > 0].mean()) if valid.sum() else None,
            iou_best_mean=float(iou_best.mean()) if has_eval else None,
            cd_median_x1e3=float(np.median(cd) * 1e3) if len(cd) else None,
            cd_mean_x1e3=float(cd.mean() * 1e3) if len(cd) else None,
            reward_mean=float(np.mean([s["score"] for s in first])),
            complete=float(np.mean([s["complete"] for s in first])),
            topology_pass=float(np.mean([s["s_topo"] >= 0.999 for s in first])),
        )
        if n > 1:
            d["best_of_n_reward"] = float(np.mean([max(s["score"] for s in r["scores"]) for r in rs]))
            d["valid_at_n"] = float(np.mean([max(s["valid_solid"] for s in r["scores"]) for r in rs]))
        return d

    out = dict(all=block(rows))
    for key in ("category", "tier"):
        groups = defaultdict(list)
        for r in rows:
            groups[f"{key}={r.get(key)}"].append(r)
        for g, rs in sorted(groups.items()):
            out[g] = block(rs)
    return out


def write_md(summary, path, name):
    cols = ["n", "invalid_ratio", "iou_mean", "iou_best_mean", "iou_mean_valid", "cd_median_x1e3", "cd_mean_x1e3",
            "reward_mean", "complete", "topology_pass"]
    with open(path, "w") as f:
        f.write(f"## {name}\n\n| subset | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for k, d in summary.items():
            vals = []
            for c in cols:
                v = d.get(c)
                vals.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
            f.write(f"| {k} | " + " | ".join(vals) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", default=os.path.join(REPO, "data", "processed", "test.jsonl"))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="debug: use GT sequences as the model output (no GPU)")
    a = ap.parse_args()
    out_dir = os.path.join(REPO, "results", a.name)
    os.makedirs(out_dir, exist_ok=True)

    if not a.merge:
        items = load_items(a.test)
        if a.limit:
            items = items[: a.limit]
        items = items[a.shard::a.num_shards]
        if a.oracle:  # pipeline check without a GPU: the "model" answers with the GT sequence
            from brepgrpo.seqfmt import strip_to_body
            man = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(os.path.dirname(a.test), "manifest.jsonl"))}
            texts = [[strip_to_body(open(man[it["id"]]["seq_path"]).read())] * a.n for it in items]
        else:
            texts = generate(items, a)
        scores = score_all(items, texts, a.workers)
        suffix = f".{a.shard}" if a.num_shards > 1 else ""
        with open(os.path.join(out_dir, f"predictions{suffix}.jsonl"), "w") as f:
            for it, ts, ss in zip(items, texts, scores):
                f.write(json.dumps(dict(id=it["id"], category=it.get("category"), tier=it.get("tier"), gt_cache=it["gt_cache"],
                                        messages=it["messages"], texts=ts, scores=ss)) + "\n")
        if a.num_shards > 1:
            return

    rows = []
    files = [f"predictions.{i}.jsonl" for i in range(a.num_shards)] if a.num_shards > 1 else ["predictions.jsonl"]
    for fn in files:
        rows += [json.loads(l) for l in open(os.path.join(out_dir, fn))]
    n = len(rows[0]["scores"]) if rows else 1
    summary = summarize(rows, n)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    write_md(summary, os.path.join(out_dir, "summary.md"), a.name)
    print(open(os.path.join(out_dir, "summary.md")).read())


if __name__ == "__main__":
    main()
