"""Fine-tune an Endor decision model on the simulator's driving decisions.

Endor (endor.thetokencompany.com) fine-tunes decision models, and this project
already emits exactly what it wants: a Choice over the seven typed pedal
decisions, a Noul for the confidence head, soft labels from the policy's own
softmax, and outcomes that are known rather than guessed at.

    pip install endor
    export ENDOR_API_KEY=edk_...          # from the Endor dashboard
    node tools/export_endor.js            # writes data/endor/*.jsonl
    python3 tools/endor_train.py

Held-out circuits are Suzuka and Shanghai, which the policy never trained on,
so `heldout` measures transfer to unseen layouts rather than a random split of
the same laps. That is a much harder and more honest evaluation.

Costs money: training is billed per GPU-second ($7.00/GPU-hour at the time of
writing, 60-second minimum per run session) and `project.evaluate` is billed as
decisions. Check Billing & limits before a long run. --dry-run prints the plan
and the row counts without contacting the API.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "endor")

PROJECT = "f1training"   # the project created in the Endor dashboard
BASE_MODEL = "decider-2b"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--checkpoint", default="v1", help="saved as <project>/<name>")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--limit", type=int, default=0,
                   help="use only the first N training rows (cheap smoke test)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="skip evaluating the base model (saves decision spend)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate the data and print the plan without calling Endor")
    p.add_argument("--check", action="store_true",
                   help="verify the key, org and balance. No training, no spend.")
    return p.parse_args()


def load_env():
    """Read KEY=value lines from a gitignored .env beside the project.

    Keeps the API key out of the shell history and out of chat. An existing
    environment variable always wins.
    """
    path = os.path.join(HERE, "..", ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read_jsonl(path):
    """Stdlib reader, so --dry-run validates the export without the SDK."""
    import json
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_local(args, use_sdk=True):
    train_path = os.path.join(DATA, "endor_train.jsonl")
    held_path = os.path.join(DATA, "endor_heldout.jsonl")
    for path in (train_path, held_path):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\nRun: node tools/export_endor.js")

    if use_sdk:
        import endor
        train, held = endor.data.load_rows(train_path), endor.data.load_rows(held_path)
    else:
        train, held = read_jsonl(train_path), read_jsonl(held_path)
    if args.limit:
        train = train[: args.limit]
    return train, held


def summarise(name, rows):
    counts, negatives = {}, 0
    for r in rows:
        labels = r.get("labels") if isinstance(r, dict) else (getattr(r, "labels", None) or {})
        probs = (labels or {}).get("pedal") or None
        if isinstance(probs, dict) and probs:
            counts[max(probs, key=probs.get)] = counts.get(max(probs, key=probs.get), 0) + 1
        if (labels or {}).get("confident") is False:
            negatives += 1
    top = ", ".join(f"{k} {100 * v / len(rows):.1f}%"
                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4])
    print(f"  {name:<9} {len(rows):>6} rows | confident=false {100 * negatives / len(rows):.1f}% | {top}")


def main():
    args = parse_args()
    load_env()
    if not args.dry_run:
        try:
            import endor
        except ImportError:
            sys.exit("pip install endor")

    train, held = load_local(args, use_sdk=not args.dry_run)
    print(f"project {args.project}  base {args.base_model}  LoRA rank {args.rank}")
    summarise("train", train)
    summarise("heldout", held)

    # One datum per labelled question, so two per row here (pedal + confident).
    datums = (train if args.dry_run else __import__("endor").data.rows_to_datums(train))
    n_datums = sum(len(r.get("labels", {})) for r in train) if args.dry_run else len(datums)
    steps = args.epochs * ((n_datums + args.batch_size - 1) // args.batch_size)
    print(f"  {n_datums} datums -> {steps} optimiser steps "
          f"({args.epochs} epochs x batch {args.batch_size}, lr {args.lr})")
    # Billed per GPU-second, so the bill scales with steps. Step latency is not
    # documented, so this brackets it rather than pretending to a number.
    for secs in (0.2, 0.5, 1.0):
        print(f"    at {secs:.1f}s/step -> {steps * secs / 3600:.1f} GPU-hours, "
              f"about ${steps * secs / 3600 * 7:.0f}")
    print("    start with --limit 2000 to measure your real step time first.")

    if args.dry_run:
        print("\ndry run: nothing sent to Endor.")
        return

    if not os.environ.get("ENDOR_API_KEY"):
        sys.exit("No ENDOR_API_KEY.\n"
                 "Put it in a gitignored .env beside this repo:\n"
                 "  printf 'ENDOR_API_KEY=edk_your_key\\n' >> ~/f1-self-driving/.env\n"
                 "or export it in your shell profile. Do not paste it into chat.")

    client = endor.EndorClient()
    who = client.whoami()
    print("signed in as:", who)
    if args.check:
        try:
            print("base models:", [b.id for b in client.base_models()])
        except Exception as exc:                      # surfaced, not swallowed
            print("could not list base models:", exc)
        try:
            print("projects:", [p_.name for p_ in client.projects.list()] or "(none yet)")
        except Exception as exc:
            print("could not list projects:", exc)
        print("\nKey works. Nothing was created and nothing was billed.")
        return
    project = client.projects.get_or_create(
        args.project, "F1 qualifying-lap pedal decisions from a physics simulator",
        kind="custom", base_models=[args.base_model])

    # Datasets are immutable, so a regenerated export needs a new name. Reuse
    # the plain name when its row count already matches, rather than minting a
    # duplicate copy of identical data on every run.
    existing = {d.name: d for d in project.datasets.list()}
    def pick(base, rows):
        d = existing.get(base)
        if d is not None and getattr(d, "n_rows", None) == len(rows):
            return base
        return base if base not in existing else f"{base}-{len(rows)}"
    train_ds, held_ds = pick("train", train), pick("heldout", held)
    for name, rows in ((train_ds, train), (held_ds, held)):
        if name in existing:
            print(f"reusing dataset {name} ({len(rows)} rows)")
        else:
            print("uploading", name)
            project.datasets.upload(name, rows)

    baseline = None
    if not args.skip_baseline:
        print(f"evaluating {args.base_model} on held-out circuits (billed as decisions)...")
        baseline = project.evaluate(f"{args.project}/{args.base_model}", held_ds).result()
        print("  baseline:", baseline.results["overall"])

    import time
    print(f"training {steps} steps...")
    t_start = time.time()
    marks: list[tuple[int, float]] = []
    with project.runs.create(base_model=args.base_model, rank=args.rank,
                             total_steps=steps, name=f"f1-{args.checkpoint}") as run:
        step = 0
        for epoch in range(args.epochs):
            for batch in endor.data.batches(datums, args.batch_size, seed=epoch):
                run.forward_backward(batch)
                run.optim_step(learning_rate=args.lr).result()
                step += 1
                if step % 25 == 0 or step == steps:
                    el = time.time() - t_start
                    # ETA off the MARGINAL rate, not the cumulative average.
                    # Run startup is a fixed few minutes; averaged over the
                    # first 25 steps it reads as 12 s/step on a 9B model and
                    # projects a 7-hour run that actually takes under two.
                    marginal = ((el - marks[-1][1]) / (step - marks[-1][0])) if marks else el / step
                    marks.append((step, el))
                    if len(marks) > 4:
                        marks.pop(0)
                    left = (steps - step) * marginal
                    print(f"  step {step}/{steps}  {marginal:.2f}s/step (avg {el/step:.2f})  "
                          f"elapsed {el/60:.1f}m  eta {left/60:.1f}m  "
                          f"spend ~${el/3600*7:.2f} of ~${(el+left)/3600*7:.2f}", flush=True)
        # save_checkpoint returns an APIFuture; without .result() the script
        # races ahead and evaluation 404s on a model that does not exist yet.
        model = run.save_checkpoint(args.checkpoint).result()
        el = time.time() - t_start
        print(f"saved {model}  ({el/60:.1f} min of GPU, about ${el/3600*7:.2f})")

    print("evaluating the fine-tune on held-out circuits...")
    tuned = project.evaluate(f"{args.project}/{args.checkpoint}", held_ds).result()
    print("  tuned:", tuned.results["overall"])
    if baseline:
        b = baseline.results["overall"].get("accuracy")
        t = tuned.results["overall"].get("accuracy")
        if b is not None and t is not None:
            print(f"\naccuracy on circuits never trained on: {b:.3f} -> {t:.3f} ({t - b:+.3f})")


if __name__ == "__main__":
    main()
