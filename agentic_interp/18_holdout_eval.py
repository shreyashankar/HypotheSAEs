"""Stage 18: generalization check — score each arm's final description on
held-out documents never seen during optimization.

The official-set F1 in report.html is optimized against: GEPA reflects on its
misclassified texts and the agent can call the scorer (and read the whole train
corpus). This stage embeds a held-out pool, runs it through the trained SAE,
and re-scores the frozen baseline / GEPA / agent descriptions there.

Held-out pools (untouched by every optimization stage):
- yelp: demo_data/yelp-demo-holdout-2K.json
- congress: 20K docs sampled from data/congress/train.json after excluding the
  training subsample (reproduced exactly via sample(20_000, random_state=0))

Two positive-sampling schemes per neuron, sharing one negative sample
(n_per_class random zero-activation held-out docs, seed=neuron):
- topk:      top n_per_class held-out docs by activation (mirrors the official
             protocol; in a smaller pool these are systematically weaker)
- threshold: held-out docs activating >= the training official set's minimum
             positive activation (random-capped at n_per_class, seed=neuron+1)

Writes {ART}/holdout_results.json and prints a per-method train->holdout summary.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)

from agentic_interp.harness import (load_artifacts, test_indices, score_description,
                                    ART, N_TEST, DATASET,
                                    AGENT_RUN_DIRNAME, GEPA_RUN_DIRNAME)
from hypothesaes.embedding import get_openai_embeddings
from hypothesaes.sae import load_model

EMBEDDER = "text-embedding-3-small"
N_CONGRESS_HOLDOUT = 20_000


def build_holdout_pool(train_texts):
    if DATASET == "yelp":
        texts = pd.read_json("demo_data/yelp-demo-holdout-2K.json", lines=True)["text"].tolist()
    else:
        full = pd.read_json("data/congress/train.json", lines=True)
        rest = full.drop(full.sample(20_000, random_state=0).index)
        texts = rest.sample(min(N_CONGRESS_HOLDOUT, len(rest)), random_state=1)["speech_text"].tolist()
    seen, pool = set(train_texts), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            pool.append(t)
    print(f"holdout pool: {len(pool)} docs ({len(texts) - len(pool)} duplicates dropped)")
    return pool


def load_arms():
    """j -> {method: {desc, train_f1}} for every arm with a finished record."""
    arms = {}
    for j_str, rec in json.load(open(os.path.join(ART, "baseline_results.json"))).items():
        arms[int(j_str)] = {"baseline": {"desc": rec["description"],
                                         "train_f1": rec["test_metrics"]["f1"]}}
    for f in glob.glob(os.path.join(ART, AGENT_RUN_DIRNAME, "n*.json")):
        rec = json.load(open(f))
        j = rec.get("neuron")
        if j in arms and rec.get("final_description"):
            arms[j]["agent"] = {"desc": rec["final_description"],
                                "train_f1": rec["official_metrics"]["f1"]}
    for f in glob.glob(os.path.join(ART, GEPA_RUN_DIRNAME, "result_n*.json")):
        rec = json.load(open(f))
        j = int(os.path.basename(f)[len("result_n"):-len(".json")])
        if j in arms:
            arms[j]["gepa"] = {"desc": rec["description"], "train_f1": rec["test_f1"]}
    return arms


ckpts = glob.glob(f"checkpoints/{DATASET}_agentic_{EMBEDDER}/*.pt")
assert len(ckpts) == 1, f"expected one checkpoint, found {ckpts}"
sae = load_model(ckpts[0], device="cpu")

texts, _, acts, _ = load_artifacts()
emb_path = os.path.join(ART, "embeddings_train.npy")
if os.path.exists(emb_path):
    emb_sample = np.load(emb_path, mmap_mode="r")[:512]
else:  # congress artifacts dropped the .npy; the on-disk embedding cache still has them
    t2e_train = get_openai_embeddings(texts[:512], model=EMBEDDER,
                                      cache_name=f"{DATASET}_agentic_{EMBEDDER}")
    emb_sample = np.stack([t2e_train[t] for t in texts[:512]])
recomputed = sae.get_activations(np.asarray(emb_sample), show_progress=False)
diff = np.abs(recomputed - acts[:512])
# Congress's stored activations came from a different embedding API fetch than
# the surviving cache; OpenAI embedding nondeterminism shifts values ~1e-3 and
# occasionally tie-flips the K-th active neuron. Require near-agreement, not
# bitwise agreement: a wrong/re-trained checkpoint disagrees on most entries.
frac_bad = float((diff > 1e-2).mean())
print(f"SAE consistency check: max diff {diff.max():.2e}, "
      f"frac entries > 1e-2: {frac_bad:.2e}")
assert frac_bad < 1e-3, "reloaded SAE disagrees with stored train activations"

hold_texts = build_holdout_pool(texts)
t2e = get_openai_embeddings(hold_texts, model=EMBEDDER,
                            cache_name=f"{DATASET}_agentic_holdout_{EMBEDDER}")
hold_emb = np.stack([t2e[t] for t in hold_texts])
hold_acts = sae.get_activations(hold_emb, show_progress=False)

arms = load_arms()
n_per_class = N_TEST // 2
results = {}
for j in sorted(arms):
    h = hold_acts[:, j]
    tp, _ = test_indices(acts, j)
    cutoff = float(acts[tp, j].min())

    rng = np.random.RandomState(j)
    zero_idx = np.where(h == 0)[0]
    neg = rng.choice(zero_idx, size=min(len(zero_idx), n_per_class), replace=False).tolist()

    n_pos_topk = int(min((h > 0).sum(), n_per_class))
    topk_pos = np.argsort(h)[-n_pos_topk:].tolist() if n_pos_topk else []
    qual = np.where(h >= cutoff)[0]
    thr_pos = (np.random.RandomState(j + 1).choice(qual, size=n_per_class, replace=False).tolist()
               if len(qual) > n_per_class else qual.tolist())

    cache = os.path.join(ART, f"annot_cache_holdout_n{j}.json")
    rec = {"cutoff": cutoff, "n_neg": len(neg), "schemes": {}}
    for scheme, pos in [("topk", topk_pos), ("threshold", thr_pos)]:
        srec = {"n_pos": len(pos),
                "pos_act_min": float(h[pos].min()) if pos else None,
                "pos_act_mean": float(h[pos].mean()) if pos else None,
                "methods": {}}
        for method, info in sorted(arms[j].items()):
            if not pos:
                continue
            m = score_description(info["desc"], pos, neg, hold_texts, h, cache_path=cache)
            srec["methods"][method] = {"holdout_f1": m["f1"],
                                       "holdout_corr": m["correlation"],
                                       "train_f1": info["train_f1"],
                                       "drop": info["train_f1"] - m["f1"]}
            print(f"n{j} {scheme:9s} {method:8s} train {info['train_f1']:.3f} -> "
                  f"holdout {m['f1']:.3f} (n_pos={len(pos)})", flush=True)
        rec["schemes"][scheme] = srec
    results[j] = rec

out_path = os.path.join(ART, "holdout_results.json")
json.dump({"dataset": DATASET, "n_per_class": n_per_class,
           "pool_size": len(hold_texts), "neurons": results},
          open(out_path, "w"), indent=2)
print(f"\nwrote {out_path}")

for scheme in ("topk", "threshold"):
    print(f"\n== {DATASET} / {scheme} ==")
    ns = [r["schemes"][scheme]["n_pos"] for r in results.values()]
    print(f"mean n_pos {np.mean(ns):.1f} (of {n_per_class})")
    for method in ("baseline", "gepa", "agent"):
        pairs = [(r["schemes"][scheme]["methods"][method]["train_f1"],
                  r["schemes"][scheme]["methods"][method]["holdout_f1"])
                 for r in results.values() if method in r["schemes"][scheme]["methods"]]
        if not pairs:
            continue
        tr, ho = np.mean([p[0] for p in pairs]), np.mean([p[1] for p in pairs])
        print(f"{method:8s} train {tr:.3f} -> holdout {ho:.3f} (drop {tr - ho:+.3f}, n={len(pairs)})")
