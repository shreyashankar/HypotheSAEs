"""Stage 19: build held-out artifacts for the blind-holdout agent experiment.

Saves {ART}/holdout_texts.json and {ART}/activations_holdout.npy: a pool of
documents outside the training corpus (so outside anything stage-10-style
agents could read), with activations from the already-trained SAE. Stage 20's
agent scores its descriptions on this pool without ever reading it.

Pools:
- yelp: demo_data/yelp-demo-holdout-2K.json (untouched by every earlier stage)
- congress: 20K docs from data/congress/train.json after excluding the training
  subsample (reproduced exactly via sample(20_000, random_state=0))
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)

from agentic_interp.harness import load_artifacts, ART, DATASET
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
print(f"SAE consistency check: max diff {diff.max():.2e}, frac entries > 1e-2: {frac_bad:.2e}")
assert frac_bad < 1e-3, "reloaded SAE disagrees with stored train activations"

hold_texts = build_holdout_pool(texts)
t2e = get_openai_embeddings(hold_texts, model=EMBEDDER,
                            cache_name=f"{DATASET}_agentic_holdout_{EMBEDDER}")
hold_emb = np.stack([t2e[t] for t in hold_texts])
hold_acts = sae.get_activations(hold_emb, show_progress=False)

json.dump(hold_texts, open(os.path.join(ART, "holdout_texts.json"), "w"))
np.save(os.path.join(ART, "activations_holdout.npy"), hold_acts)
alive = (hold_acts.max(axis=0) > 0).sum()
print(f"Done. Holdout activations {hold_acts.shape}, alive neurons: {alive}/256")
