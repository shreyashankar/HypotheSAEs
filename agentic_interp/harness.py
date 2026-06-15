"""Shared harness: artifact loading, neuron selection, official-set scoring.

`test_indices` exactly replicates the repo's scoring protocol
(sample_top_zero with random_seed=neuron_idx, n_examples = N_TEST).
"""
import os, sys, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from hypothesaes.annotate import annotate
from hypothesaes.utils import truncate_text

DATASET = os.environ.get("AGENTIC_DATASET", "yelp")
ART = os.environ.get("AGENTIC_ART",
                     os.path.join(REPO, "agentic_interp",
                                  "artifacts" if DATASET == "yelp" else f"artifacts_{DATASET}"))
ANNOTATOR_MODEL = "gpt-5-mini"
MAX_WORDS = 256
# Official protocol size: Yelp follows the quickstart default (50 top + 50 zero);
# Congress follows the paper reproduction notebook (100 top + 100 zero).
N_TEST = 200 if DATASET == "congress" else 100

_INSTRUCTIONS = {
    "yelp": """All of the texts are reviews of restaurants on Yelp.
Features should describe a specific aspect of the review. For example:
- "mentions long wait times to receive service"
- "praises how a dish was cooked, with phrases like 'perfect medium-rare'\"""",
    "congress": """All of the texts are excerpts of speeches from the US Congress.
Example features may include:
- "mentions that the economy is strong"
- "discusses environmental issues or environmental policy\"""",
}
_DESCS = {
    "yelp": "10,000 Yelp restaurant reviews",
    "congress": "20,000 ten-sentence excerpts of US Congressional speeches",
}
TASK_SPECIFIC_INSTRUCTIONS = _INSTRUCTIONS[DATASET]
DATASET_DESC = _DESCS[DATASET]

# Per-neuron run-record directory names (basenames under each dataset's ART).
# Defined once so writers (10/11) and readers (07/14/16) agree by construction.
AGENT_RUN_DIRNAME = os.environ.get("AGENT_RUN_NAME", "agent_runs_strict")
GEPA_RUN_DIRNAME = "gepa_runs_v5"


def load_artifacts(with_emb=False):
    """Load texts, stars, activations. Embeddings (large) only when with_emb."""
    acts = np.load(os.path.join(ART, "activations_train.npy"))
    recs = json.load(open(os.path.join(ART, "train_texts.json")))
    texts = [r["text"] for r in recs]
    stars = np.array([r.get("stars", r.get("label")) for r in recs], dtype=float)
    emb = np.load(os.path.join(ART, "embeddings_train.npy")) if with_emb else None
    return texts, stars, acts, emb


def select_neurons(acts, stars, n_predictive=10, n_random=10, seed=0):
    """Top-|corr with stars| neurons + random alive neurons (disjoint)."""
    alive = np.where(acts.max(axis=0) > 0)[0]
    corrs = np.zeros(acts.shape[1])
    for j in alive:
        a = acts[:, j]
        if a.std() > 0:
            corrs[j] = abs(np.corrcoef(a, stars)[0, 1])
    predictive = list(np.argsort(-corrs)[:n_predictive])
    rng = np.random.RandomState(seed)
    pool = [j for j in alive if j not in predictive]
    random_picks = list(rng.choice(pool, size=n_random, replace=False))
    return [int(j) for j in predictive + random_picks]


def test_indices(acts, neuron_idx, n_test=None):
    """Replicates sample_top_zero(n_examples, random_seed=neuron_idx) index logic.

    n_test defaults to the env-derived N_TEST; pass explicitly when one process
    handles multiple datasets (e.g. the report builder).
    """
    neuron_acts = acts[:, neuron_idx]
    n_per_class = (N_TEST if n_test is None else n_test) // 2
    np.random.seed(neuron_idx)
    n_pos = min(int(np.sum(neuron_acts > 0)), n_per_class)
    top_idx = np.argsort(neuron_acts)[-n_pos:]
    zero_idx = np.where(neuron_acts == 0)[0]
    rand_idx = np.random.choice(zero_idx, size=min(len(zero_idx), n_per_class), replace=False)
    return top_idx.tolist(), rand_idx.tolist()


def compute_metrics(annotations, labels, activations):
    """Identical math to NeuronInterpreter._compute_metrics."""
    annotations = np.asarray(annotations).astype(bool)
    labels = np.asarray(labels).astype(bool)
    if not (labels.any() and (~labels).any()):
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}
    tp = np.sum(annotations & labels)
    fp = np.sum(annotations & ~labels)
    fn = np.sum(~annotations & labels)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    correlation = float(np.corrcoef(np.asarray(activations, dtype=float), annotations)[0, 1]) \
        if len(np.unique(annotations)) > 1 else 0.0
    return {"recall": float(recall), "precision": float(precision),
            "f1": float(f1), "correlation": correlation}


def score_description(description, pos_idx, neg_idx, texts, neuron_acts,
                      cache_path=None, n_workers=20, return_errors=False):
    """Official fidelity scoring: repo annotate prompt + gpt-5-mini annotator."""
    idx = list(pos_idx) + list(neg_idx)
    eval_texts = [truncate_text(texts[i], MAX_WORDS) for i in idx]
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    activations = neuron_acts[idx]
    tasks = [(t, description) for t in eval_texts]
    results = annotate(tasks=tasks, model=ANNOTATOR_MODEL, cache_path=cache_path,
                       n_workers=n_workers, show_progress=False)
    ann = np.array([results[description].get(t, 0) for t in eval_texts])
    metrics = compute_metrics(ann, labels, activations)
    if return_errors:
        fps = [int(idx[i]) for i in range(len(idx)) if ann[i] == 1 and labels[i] == 0]
        fns = [int(idx[i]) for i in range(len(idx)) if ann[i] == 0 and labels[i] == 1]
        metrics["false_positive_idx"] = fps
        metrics["false_negative_idx"] = fns
    return metrics
