"""Compare Endor models on the held-out circuits, against each other and
against the baselines that actually matter.

Reads the JSON that tools/endor_eval.py saved, so it costs nothing to re-run.

    ./.venv/bin/python tools/endor_report.py decider-2b v1 jev-9b jev-v1

A fine-tune's accuracy means little on its own here. The pedal classes are
heavily imbalanced -- a qualifying lap is mostly full throttle -- and the
confidence question is 98 % true, so a model that always answers "yes" scores
0.98. Both trivial baselines are computed from the held-out file and printed
alongside, because a number that loses to a constant predictor should look like
it loses.
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "endor")
HELD = os.path.join(DATA, "endor_heldout.jsonl")


def trivial_baselines() -> dict:
    rows = [json.loads(l) for l in open(HELD)]
    n = len(rows)
    pedal = collections.Counter()
    yes = 0
    for r in rows:
        p = r["labels"]["pedal"]
        pedal[max(p, key=p.get)] += 1
        yes += 1 if r["labels"]["confident"] else 0
    top, top_n = pedal.most_common(1)[0]
    rate = yes / n
    return {
        "n": n,
        "choice": {"name": f"always {top}", "accuracy": top_n / n},
        # Always-true at p=1.0: Brier is just the miss rate, ECE the gap to it.
        "noul": {"name": "always true", "accuracy": rate,
                 "brier": 1 - rate, "ece": 1 - rate},
    }


def load(model: str) -> dict | None:
    path = os.path.join(DATA, f"eval_f1training_{model}.json")
    if not os.path.exists(path):
        return None
    raw = json.load(open(path))["raw"]
    res = raw.get("results", {})
    by = res.get("by_type", {})
    return {"overall": res.get("overall", {}), "choice": by.get("choice", {}),
            "noul": by.get("noul", {})}


def cell(d: dict, key: str) -> str:
    v = d.get(key)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "   -  "


def main() -> None:
    models = sys.argv[1:] or ["decider-2b", "v1", "jev-9b", "jev-v1"]
    base = trivial_baselines()
    found = [(m, load(m)) for m in models]
    missing = [m for m, d in found if d is None]
    found = [(m, d) for m, d in found if d]
    if not found:
        sys.exit("no evaluations found. Run: ./.venv/bin/python tools/endor_eval.py f1training/<model>")

    print(f"Held-out circuits (Suzuka + Shanghai), {base['n']} rows. "
          "Neither the policy nor these models has driven them.\n")

    for section, label in (("choice", "CHOICE  which pedal input"),
                           ("noul", "NOUL  will this turn out well")):
        print(f"{label}")
        w = 20
        print(f"  {'model':<{w}}{'accuracy':>10}{'brier':>9}{'ece':>8}{'conf':>8}")
        b = base[section]
        print(f"  {b['name']:<{w}}{b['accuracy']:>10.3f}{cell(b,'brier'):>9}"
              f"{cell(b,'ece'):>8}{'   -  ':>8}   <- trivial baseline")
        for m, d in found:
            s_ = d[section]
            print(f"  {m:<{w}}{cell(s_,'accuracy'):>10}{cell(s_,'brier'):>9}"
                  f"{cell(s_,'ece'):>8}{cell(s_,'mean_confidence'):>8}")
        print()

    print("OVERALL")
    print(f"  {'model':<20}{'accuracy':>10}{'brier':>9}{'ece':>8}")
    for m, d in found:
        o = d["overall"]
        print(f"  {m:<20}{cell(o,'accuracy'):>10}{cell(o,'brier'):>9}{cell(o,'ece'):>8}")

    print("\nSELECTIVE  accuracy when the model is confident, and how often it is")
    print(f"  {'model':<20}{'>=0.90 acc':>12}{'cover':>8}{'>=0.95 acc':>12}{'cover':>8}")
    for m, d in found:
        sel = (d["overall"].get("selective") or {})
        def g(t, k):
            v = (sel.get(t) or {}).get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) else "   -  "
        print(f"  {m:<20}{g('0.9','accuracy'):>12}{g('0.9','coverage'):>8}"
              f"{g('0.95','accuracy'):>12}{g('0.95','coverage'):>8}")

    if missing:
        print("\nnot evaluated yet:", ", ".join(missing))


if __name__ == "__main__":
    main()
