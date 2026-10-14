"""Ask the Endor model what it would do, at points the local policy drives.

This is where the typed client in endor_client.py actually gets used. It is
deliberately NOT in the driving loop: the controller runs at 60 Hz and
decisions are billed per call, so a remote API cannot sit inside it. What it
can do is give a second opinion offline -- sample states from a lap, ask the
fine-tuned model, and see where it disagrees with the evolved network.

Disagreements are the interesting part. They are the states where distillation
did not carry over, and they cluster somewhere specific on a circuit rather
than scattering evenly.

    ./.venv/bin/python tools/endor_compare.py f1training/v1 --circuit suzuka -n 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from endor_client import Choice, EndorClient, EndorError, Noul  # noqa: E402

DECISIONS = ["HARD_BRAKE", "BRAKE", "TRAIL_BRAKE", "COAST", "FEATHER", "MODULATE", "FULL_POWER"]


def sample_states(circuit: str, n: int) -> list[dict]:
    """Reuse the exporter so these states are identical to the training ones."""
    path = os.path.join(HERE, "..", "data", "endor")
    rows = []
    for name in ("endor_heldout.jsonl", "endor_train.jsonl"):
        f = os.path.join(path, name)
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"].startswith(f"{circuit}-clean-"):
                    rows.append(r)
    if not rows:
        sys.exit(f"no exported rows for {circuit!r}. Run: node tools/export_endor.js")
    rows.sort(key=lambda r: r["id"])
    stride = max(1, len(rows) // n)
    return rows[::stride][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="f1training/v1")
    ap.add_argument("--circuit", default="suzuka")
    ap.add_argument("-n", type=int, default=30, help="states to sample")
    args = ap.parse_args()

    rows = sample_states(args.circuit, args.n)
    client = EndorClient(model=args.model)
    questions = {
        "pedal": Choice(rows[0]["questions"]["pedal"]["instructions"],
                        rows[0]["questions"]["pedal"]["criteria"]),
        "confident": Noul(rows[0]["questions"]["confident"]["instructions"]),
    }

    print(f"{args.model} vs the local policy on {args.circuit}, {len(rows)} states")
    print("(the first call can take minutes if the model is cold)\n")
    agree, tokens, disagreements = 0, 0, []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        local_probs = row["labels"]["pedal"]
        local = max(local_probs, key=local_probs.get)
        try:
            ans = client.decide(row["state"], questions)
        except EndorError as exc:
            sys.exit(f"\n{exc}")
        remote, p = ans.choice("pedal")
        conf = ans.noul("confident")
        tokens += ans.input_tokens
        same = remote == local
        agree += same
        if not same:
            disagreements.append((row["id"], local, remote, p))
        if i == 1 or i % 10 == 0 or not same:
            flag = "    " if same else " <- "
            print(f"  {i:>3}/{len(rows)} {row['id']:<22} local {local:<11}"
                  f"{flag}endor {remote:<11} p={p:.2f} conf={conf:.2f}")

    el = time.time() - t0
    print(f"\nagreement {agree}/{len(rows)} ({100 * agree / len(rows):.0f}%)  "
          f"{tokens} input tokens  {el:.0f}s  {el / len(rows):.1f}s per decision")
    if disagreements:
        by = collections.Counter((d[1], d[2]) for d in disagreements)
        print("\nwhere they differ (local -> endor):")
        for (a, b), c in by.most_common():
            print(f"  {a:<12} -> {b:<12} {c}x")
        print("\nAt 60 Hz a lap needs ~5,000 decisions, so this is a second "
              "opinion offline, not a driver.")


if __name__ == "__main__":
    main()
