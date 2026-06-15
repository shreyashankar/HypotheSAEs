"""Re-score existing baseline descriptions on the current official protocol set
(needed after switching Congress to the 200-example protocol). Archives the old
file the first time."""
import os, sys, json, shutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_interp.harness import (load_artifacts, test_indices, score_description,
                                    ART, N_TEST)

texts, _, acts, _ = load_artifacts()
path = os.path.join(ART, "baseline_results.json")
baseline = json.load(open(path))
archive = path.replace(".json", "_n100.json")
if not os.path.exists(archive):
    shutil.copy(path, archive)

for j, rec in baseline.items():
    tp, tn = test_indices(acts, int(j))
    cache = os.path.join(ART, f"annot_cache_test_n{j}.json")
    m = score_description(rec["description"], tp, tn, texts, acts[:, int(j)], cache_path=cache)
    rec["test_metrics"] = m
    print(f"n{j}: {m['correlation']:.3f} (n={len(tp)+len(tn)})", flush=True)

json.dump(baseline, open(path, "w"), indent=2)
cs = [r["test_metrics"]["correlation"] for r in baseline.values()]
fs = [r["test_metrics"]["f1"] for r in baseline.values()]
print(f"BASELINE on N_TEST={N_TEST}: mean corr {np.mean(cs):.3f}, mean f1 {np.mean(fs):.3f}")
