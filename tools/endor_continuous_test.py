"""Try Endor continuous learning on the F1 decisions.

    .venv/bin/python tools/endor_continuous_test.py create
    .venv/bin/python tools/endor_continuous_test.py eval  <tag> [n]
    .venv/bin/python tools/endor_continuous_test.py feed  <n> [--corrections]
    .venv/bin/python tools/endor_continuous_test.py status

Stops when the balance falls more than MAX_SPEND dollars under where it began.
"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from endor_train import load_env  # noqa: E402
import endor  # noqa: E402

load_env()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "endor")
NAME, BASE, MAX_SPEND = "f1-continuous", "mapika/decider-2b", 6.0
STATE = os.path.join(ROOT, "continuous_test.json")
c = endor.EndorClient()


def st():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"start_balance": float(c.balance()), "evals": {}, "fed": 0}


def save(s):
    json.dump(s, open(STATE, "w"), indent=2, default=str)


def guard(s):
    spent = s["start_balance"] - float(c.balance())
    if spent > MAX_SPEND:
        raise SystemExit(f"budget guard: spent ${spent:.2f}")
    return spent


def rows(path, n, seed):
    allr = [json.loads(l) for l in open(os.path.join(ROOT, path))]
    random.Random(seed).shuffle(allr)
    return allr[:n]


def decide(r):
    return c.system_one(r["state"], r["questions"], model=NAME)


def dump(res):
    d = res.model_dump() if hasattr(res, "model_dump") else res
    return d["answers"] if "answers" in d else d


def main():
    cmd = sys.argv[1]
    s = st(); save(s)
    p = c.projects.get_or_create(NAME, base_model=BASE, description="F1 pedal decisions from simulator state") if cmd == "create" else c.projects.get(NAME)
    if cmd == "create":
        print("project ready:", NAME, "base", BASE)
    elif cmd == "eval":
        tag, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 150
        model = NAME if tag != "preview" else NAME + ":preview"
        data = rows("endor_heldout.jsonl", n, 7)
        def one(r):
            res = c.system_one(r["state"], r["questions"], model=model)
            a = dump(res)
            pick = a["pedal"]["choice"]; conf = float(a["confident"]["noul"])
            return pick == max(r["labels"]["pedal"], key=r["labels"]["pedal"].get), (conf > .5) == r["labels"]["confident"]
        with ThreadPoolExecutor(8) as ex:
            out = list(ex.map(one, data))
        res = {"model": model, "n": len(out), "pedal_acc": sum(o[0] for o in out) / len(out), "conf_acc": sum(o[1] for o in out) / len(out)}
        s["evals"][tag] = res; print(res)
    elif cmd == "feed":
        n = int(sys.argv[2]); corr = "--corrections" in sys.argv
        data = rows("endor_train.jsonl", n, 11 + s["fed"])
        def one(r):
            res = decide(r)
            if corr:
                a = dump(res); truth = max(r["labels"]["pedal"], key=r["labels"]["pedal"].get)
                c.corrections.create(res, {"pedal": {"correct": a["pedal"]["choice"] == truth, "answer": truth},
                                           "confident": {"correct": (float(a["confident"]["noul"]) > .5) == r["labels"]["confident"]}})
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, data))
        s["fed"] += n
    if cmd in ("status", "feed", "eval", "create"):
        print("learning:", p.learning() if hasattr(p, "learning") else "n/a")
        try: print("versions:", [(v.name, v.production, v.preview, v.accuracy) for v in p.versions()])
        except Exception as e: print("versions err", e)
    print(f"spent so far: ${guard(s):.2f}"); save(s)


main()
