# Agentic optimization of SAE feature descriptions

Can an LLM agent write better natural-language descriptions of SAE features than
the repo's one-shot interpreter? This compares three methods on the repo's own
fidelity protocol, on two datasets from the paper.

**Open `report.html` and `transcripts.html` in a browser** — they are
self-contained (data embedded), so no server or artifacts are needed to view them.

## The three methods

All three describe a neuron with one sentence, then are scored identically: an
annotator LLM (`gpt-5-mini`, the repo's `annotate` prompt) answers yes/no "does
this TEXT satisfy this PROPERTY?" on the neuron's official set — its top-K
activating texts + K random zero-activation texts — and the score is the **F1**
of those answers vs. the labels (the repo's primary metric; 1.0 = perfect
discrimination). All three are seeded with the baseline description as a floor.

1. **Baseline** — the repo's default `NeuronInterpreter`: one `gpt-5.2` call over
   the top-10 + 10 random examples. No iteration.
2. **GEPA** — vanilla [`gepa`](https://github.com/gepa-ai/gepa): evolves the
   description by reflecting (`gpt-5.5`) on misclassified texts. Sees scores and
   error examples only — no corpus access.
3. **Agent** — one OpenAI Agents-SDK agent (`gpt-5.5`) per neuron with a
   persistent Python sandbox over the full corpus/activations/embeddings **and**
   the official scorer (callable on the official set or any sample it picks).

GEPA and the agent get the same 1,000-annotator-call budget and the same
single-sentence output constraint. The only difference is the sandbox.

## Headline result (official-set F1, 20 neurons/dataset)

| dataset | baseline | GEPA | agent |
|---|---|---|---|
| Yelp (review → star rating) | 0.755 | 0.888 | **0.962** |
| Congress (speech → party) | 0.628 | 0.878 | **0.942** |

Agent vs GEPA head-to-head: 17–2 (Yelp), 14–2 (Congress). Fresh-annotation
re-scoring (cache bypassed) moved means by ≤0.004 — the scores are real, not
caching/selection artifacts.

**What the sandbox buys:** corpus-wide probes crack lexical/disjunctive neurons
that error-reflection alone stalls on — e.g. `grill`-substring, "Pat's vs Geno's
cheesesteaks", multi-cuisine disjunctions, "business name starts with B". On
clean concept neurons GEPA matches the agent and is more budget-efficient early.

**Caveat learned the hard way:** with pure F1 hill-climbing, incoherent neurons
get "described" by enumerating their eval texts. The single-sentence-property
constraint curbs this; where a neuron still can't be captured honestly (e.g.
Congress n79 ≈ 0.77), that gap is a signal the neuron isn't a coherent concept.

## Agent with held-out scoring (stages 19-20)

The methods above optimize against an evaluation set they can read, so a
description can score well by fitting those specific texts. Stage 20 reruns the
agent with its official score computed on a held-out set of documents that the
agent can score but never read (2,000 unseen Yelp reviews or 20,000 unseen
Congress chunks; the evaluation set is the pool's top K activating documents
plus K random documents with zero activation). Scoring returns the overall
precision, recall, and F1 and nothing else. The sandbox, budget, and output
constraints are unchanged.

| held-out F1 (20 neurons) | baseline | GEPA | agent | agent with held-out scoring |
|---|---|---|---|---|
| Yelp | 0.625 | 0.775 | 0.816 | **0.860** |
| Congress | 0.601 | 0.879 | 0.929 | **0.913** |

The first three columns take the descriptions from the table above and score
them on the held-out set. The agent with held-out scoring beats the original
agent's descriptions on Yelp (0.860 vs 0.816) and roughly matches them on
Congress (0.913 vs 0.929). Its scores on the training set and the held-out set
are nearly equal (Congress 0.913 and 0.913, Yelp 0.888 and 0.860), so its
descriptions are not fitted to any particular evaluation set. The cost is a
lower score on the readable training set (0.888 and 0.913, vs 0.962 and 0.942
for the original agent). A full held-out scoring spends 100 to 200 of the
1,000-call budget, and the agent cannot see which texts it got wrong, so it
tries fewer, more distinct candidates instead of fine-tuning wording.

## Pipeline (numbered stages, run in order)

Dataset is selected by the `AGENTIC_DATASET` env var (`yelp` default, or
`congress`); set `OPENAI_KEY_SAE`. Shared logic lives in `harness.py`.

| stage | what it does |
|---|---|
| `01_embed_train_sae.py` | embed Yelp demo reviews, train the Top-K SAE, cache activations |
| `05_congress_prep.py` | same for a 20K Congress subset |
| `02_baseline.py` | repo-default descriptions for the 20 studied neurons |
| `13_rescore_baseline.py` | re-score the baseline on Congress's 200-example protocol |
| `10_agent_v5.py` | the agent method (writes per-neuron records + transcripts) |
| `11_gepa_v5.py` | vanilla GEPA (shardable via `GEPA_SHARD=i/n`) |
| `14_verify.py` | integrity check: re-score every winner with the cache bypassed |
| `19_prep_holdout.py` | build the held-out pool (texts + SAE activations) |
| `20_agent_holdout.py` | rerun the agent with its official score computed on the held-out set, which it can score but never read |
| `07_make_report.py` | build `report.html` |
| `16_build_transcript_viewer.py` | build `transcripts.html` |

`watch_report.sh` rebuilds both HTML files as records land during a run.

Artifacts (embeddings, activations, annotation caches, per-neuron records) are
git-ignored and regenerated by the stages above; only source + the two HTML
deliverables are tracked.
