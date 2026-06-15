"""Integrity check: re-score every method's final description with COMPLETELY
FRESH annotations (no cache — defeats any hypothetical cache tampering and
quantifies annotator stochasticity). Run after the fleets finish."""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_interp.harness import (load_artifacts, test_indices, score_description, ART,
                                    AGENT_RUN_DIRNAME, GEPA_RUN_DIRNAME)

texts, _, acts, _ = load_artifacts()


def fresh_score(desc, j):
    """Official scoring with the cache bypassed — every annotation is a fresh call."""
    tp, tn = test_indices(acts, j)
    return score_description(desc, tp, tn, texts, acts[:, j], cache_path=None, n_workers=60)


out = {}
# agent
for f in sorted(glob.glob(os.path.join(ART, AGENT_RUN_DIRNAME, "n*.json"))):
    rec = json.load(open(f))
    j = rec["neuron"]
    m = fresh_score(rec["final_description"], j)
    out.setdefault("agent", {})[str(j)] = {
        "claimed_f1": rec["official_metrics"]["f1"], "fresh_f1": m["f1"]}
    print(f"agent n{j}: claimed F1 {rec['official_metrics']['f1']:.3f} -> fresh {m['f1']:.3f}", flush=True)
# gepa
for f in sorted(glob.glob(os.path.join(ART, GEPA_RUN_DIRNAME, "result_n*.json"))):
    rec = json.load(open(f))
    j = int(os.path.basename(f)[len("result_n"):-len(".json")])
    m = fresh_score(rec["description"], j)
    out.setdefault("gepa", {})[str(j)] = {
        "claimed_f1": rec["test_f1"], "fresh_f1": m["f1"]}
    print(f"gepa n{j}: claimed F1 {rec['test_f1']:.3f} -> fresh {m['f1']:.3f}", flush=True)

json.dump(out, open(os.path.join(ART, "verification.json"), "w"), indent=2)
for method, d in out.items():
    gaps = [v["claimed_f1"] - v["fresh_f1"] for v in d.values()]
    print(f"{method}: mean claimed-vs-fresh F1 gap {np.mean(gaps):+.4f} (n={len(gaps)})")
