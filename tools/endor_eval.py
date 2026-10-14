"""Evaluate an Endor model on the held-out circuits and save the result.

    ./.venv/bin/python tools/endor_eval.py f1training/decider-2b [dataset]

Billed as decisions, which are cheap next to GPU time. Results land in
data/endor/eval_<model>.json so runs can be compared later without paying
for the evaluation twice.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from endor_train import load_env  # noqa: E402

import endor  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "endor")


def flat(results, key):
    d = results.get(key) or {}
    return {k: d.get(k) for k in ("n", "accuracy", "brier", "nll", "ece", "mean_confidence")}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "f1training/decider-2b"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "heldout"
    load_env()
    project = model.split("/")[0]
    p = endor.EndorClient().projects.get(project)

    t0 = time.time()
    print(f"evaluating {model} on {dataset} (a cold start can take minutes)...", flush=True)
    ev = p.evaluate(model, dataset).result()
    d = ev.model_dump() if hasattr(ev, "model_dump") else vars(ev)
    elapsed = time.time() - t0

    res = d.get("results") or {}
    by = res.get("by_type") or {k: v for k, v in res.items() if k in ("choice", "noul")}
    record = {
        "model": model, "dataset": dataset, "seconds": round(elapsed, 1),
        "overall": flat(res, "overall"),
        "choice": flat(by, "choice"),
        "noul": flat(by, "noul"),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"eval_{model.replace('/', '_')}.json")
    with open(path, "w") as fh:
        json.dump({"record": record, "raw": d}, fh, indent=2, default=str)

    print(f"done in {elapsed:.0f}s -> {path}\n")
    print(f"{'':10}{'n':>7}{'accuracy':>10}{'brier':>9}{'ece':>8}{'conf':>8}")
    for name in ("overall", "choice", "noul"):
        r = record[name]
        if r.get("n") is None:
            continue
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "-"
        print(f"{name:10}{r['n']:>7}{fmt(r['accuracy']):>10}{fmt(r['brier']):>9}"
              f"{fmt(r['ece']):>8}{fmt(r['mean_confidence']):>8}")


if __name__ == "__main__":
    main()
