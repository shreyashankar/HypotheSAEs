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

## Blind-holdout variant (stages 19–20)

Does the agent's edge survive when the scored texts cannot be memorized? Stage
20 reruns the agent with its official metric moved to a **held-out corpus it
can score but never read** (2K unseen Yelp reviews / 20K unseen Congress
chunks; official set = the pool's top-K activating + K seeded zero-activation
docs). Scoring it returns aggregate P/R/F1 only — no texts, no misclassified
examples. Sandbox, budget, and constraints are otherwise identical to stage 10.

| held-out F1 (20 neurons) | baseline (frozen) | GEPA (frozen) | agent (frozen) | blind-holdout agent |
|---|---|---|---|---|
| Yelp | 0.625 | 0.775 | 0.816 | **0.860** |
| Congress | 0.601 | 0.879 | 0.929 | **0.913** |

"Frozen" = the train-optimized descriptions above, re-scored on the same
held-out sets. The blind agent matches (Congress, −0.016) or beats (Yelp,
+0.044) the transferred train-optimized labels, and its own train↔held-out gap
is ~0 (Congress 0.913/0.913; Yelp 0.888 train / 0.860 held-out) — the labels it
writes are honest concepts, not eval-set artifacts. Cost of blindness: it
scores lower on the readable train set (0.888/0.913 vs 0.962/0.942), since a
full blind scoring costs 100–200 of the 1,000-call budget and small-variation
hill-climbing is pointless without visible errors.

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
| `19_prep_holdout.py` | build the held-out pool (texts + SAE activations) for the blind-holdout experiment |
| `20_agent_holdout.py` | the blind-holdout agent (scores the held-out set, can never read it) |
| `07_make_report.py` | build `report.html` |
| `16_build_transcript_viewer.py` | build `transcripts.html` |

`watch_report.sh` rebuilds both HTML files as records land during a run.

Artifacts (embeddings, activations, annotation caches, per-neuron records) are
git-ignored and regenerated by the stages above; only source + the two HTML
deliverables are tracked.
