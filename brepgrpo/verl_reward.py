"""verl entry point:  reward.custom_reward_function.path=<repo>/brepgrpo/verl_reward.py
                    reward.custom_reward_function.name=compute_score

verl's `remote` reward manager runs this in a pool of Ray actors (one process each).
The OCC part (compile, mesh, compare) runs in a child process of that actor with a
hard timeout (pebble), because OpenCascade can hang or segfault on a malformed
sequence: such a sample keeps its cheap topology reward, is flagged timeout=1, and the
training step goes on. Without the child, a hang would stall the step and a segfault
would kill the Ray actor and with it the run.

BREP_REWARD_TIMEOUT   seconds per sample before the child is killed (default 45)
BREP_REWARD_INPROCESS set to 1 to score in-process without the child (debugging only)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import TimeoutError as FutTimeout

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from brepgrpo.reward import cheap_part, full_part  # noqa: E402

TIMEOUT_S = float(os.environ.get("BREP_REWARD_TIMEOUT", "45"))
INPROCESS = os.environ.get("BREP_REWARD_INPROCESS", "0") == "1"
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from pebble import ProcessPool
        # one child per actor; recycled every 500 samples so OCC memory cannot creep
        _POOL = ProcessPool(max_workers=1, max_tasks=500, context=mp.get_context("spawn"))
    return _POOL


def _run_isolated(seq_text, gt_path, partial):
    global _POOL
    try:
        return _pool().schedule(full_part, args=(seq_text, gt_path, partial), timeout=TIMEOUT_S).result()
    except FutTimeout:
        pass                                    # pebble has already killed and replaced the child
    except RuntimeError as ex:                  # the pool itself is broken: rebuild it on the next sample
        print(f"[brep reward] pool error, rebuilding: {ex}", file=sys.stderr, flush=True)
        _POOL = None
    except Exception as ex:                     # child crashed (ProcessExpired) or raised; pebble replaces it
        print(f"[brep reward] {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
    out = dict(partial)
    out["timeout"] = 1.0
    return out


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    seq, partial, seq_text = cheap_part(solution_str or "")
    try:                                        # curriculum tier of the prompt (prepare_data.py), for the training log
        partial["tier"] = float((extra_info or {}).get("tier", -1))
    except (TypeError, ValueError, AttributeError):
        partial["tier"] = -1.0
    if seq is None:                             # truncated / unparseable: reward 0, no OCC needed
        return partial
    if INPROCESS:
        return full_part(seq_text, gt["cache"], partial)
    return _run_isolated(seq_text, gt["cache"], partial)
