"""Pre-flight on Nova (CPU is enough, ~1-2 min). Catches env/data problems before GPU hours.

  python tests/preflight_nova.py --data data/processed --model Qwen/Qwen2.5-VL-3B-Instruct

1. verl's own SFT dataset tokenizes our parquet with the real chat template, loss only on the answer
2. RL prompts fit data.max_prompt_length (1280)
3. the verl reward entry point scores GT sequences ~1.0 in this env (OpenCascade + pebble present)
4. the token-length estimate prepare_data.py used for the length filters vs the real tokenizer
"""
import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "processed"))
    ap.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()

    import pandas as pd
    from transformers import AutoProcessor, AutoTokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        proc = AutoProcessor.from_pretrained(a.model)
    except Exception:
        proc = None
    sub = os.path.join("/tmp", f"preflight_sft_{os.getpid()}.parquet")
    pd.read_parquet(os.path.join(a.data, "sft_train.parquet")).head(a.n).to_parquet(sub)
    cfg = dict(max_length=16384, truncation="right", pad_mode="no_padding", messages_key="messages")
    ds = MultiTurnSFTDataset(sub, tok, cfg, processor=proc)
    lens = []
    for i in range(len(ds)):
        item = ds[i]
        ids, lm = item["input_ids"], item["loss_mask"]
        assert lm.sum() > 0, "empty loss mask"
        ans = tok.decode(ids[lm.bool()])
        assert "end_model()" in ans, "answer tokens do not contain end_model()"
        assert "Construction:" not in ans and "You are a CAD engineer" not in ans, "loss leaks into the prompt"
        lens.append(int(ids.shape[0]))
    print(f"[1] SFT dataset OK: {len(ds)} samples, tokens min/max {min(lens)}/{max(lens)}, loss only on the answer")

    rl = pd.read_parquet(os.path.join(a.data, "rl_train_8k.parquet"))
    def n_tok(msgs):
        out = tok.apply_chat_template([dict(m) for m in msgs], tokenize=True, add_generation_prompt=True)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)

    pl = [n_tok(p) for p in rl["prompt"].head(200)]
    print(f"[2] RL prompt tokens (first 200): max {max(pl)} (limit 1280) -> {'OK' if max(pl) <= 1280 else 'TOO LONG'}")

    spec = importlib.util.spec_from_file_location("vr", os.path.join(REPO, "brepgrpo", "verl_reward.py"))
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    from brepgrpo.seqfmt import strip_to_body
    rows = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))]
    rows = [r for r in rows if r["gt_ok"]][:5]
    sc = [vr.compute_score("x", strip_to_body(open(r["seq_path"]).read()), json.dumps(dict(id=r["id"], cache=r["gt_cache"])), {})["score"] for r in rows]
    print(f"[3] reward on GT: {[round(s, 3) for s in sc]} -> {'OK' if min(sc) > 0.99 else 'CHECK'}")

    # the estimate must not undercount, or sequences slip past the 8k / 16k length filters.
    # Loaded by file path: verl's editable install puts its own `scripts` package on sys.path, which shadows ours.
    spec = importlib.util.spec_from_file_location("prepare_data", os.path.join(REPO, "scripts", "prepare_data.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    est_tokens = prep.est_tokens
    rows = [json.loads(l) for l in open(os.path.join(a.data, "manifest.jsonl"))][:300]
    ratio = []
    for r in rows:
        body = strip_to_body(open(r["seq_path"]).read())
        ratio.append(est_tokens(body) / max(1, len(tok(body)["input_ids"])))
    print(f"[4] est_tokens / real tokens on 300 GT bodies: min {min(ratio):.3f} mean {sum(ratio) / len(ratio):.3f} "
          f"max {max(ratio):.3f} -> {'OK (never undercounts)' if min(ratio) >= 1.0 else 'UNDERCOUNTS: raise the margin in est_tokens'}")


if __name__ == "__main__":
    main()
