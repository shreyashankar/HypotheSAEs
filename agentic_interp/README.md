# Agentic optimization of SAE feature descriptions

Can an LLM agent write better natural language descriptions of SAE features
than the repo's default interpreter, which makes a single LLM call? This
experiment compares four methods on the repo's own scoring protocol, on two
datasets from the paper.

**Open `report.html` and `transcripts.html` in a browser.** Both files have the
data embedded, so you do not need a server or the artifacts to view them.

## How descriptions are scored

Each method describes a neuron with one sentence, and every description is
scored the same way. An annotator LLM (`gpt-5-mini`, using the repo's
`annotate` prompt) reads one text and one description at a time and answers yes
or no to "does this TEXT satisfy this PROPERTY?". The evaluation set for a
neuron is its K texts with the highest activation (labeled positive) plus K
random texts where the neuron does not activate (labeled negative), with K = 50
on Yelp and K = 100 on Congress. The score is the F1 of the annotator's answers
against those labels, so 1.0 means the description perfectly separates
activating texts from non-activating texts.

## The four methods

1. **Baseline.** The repo's default `NeuronInterpreter`. One `gpt-5.2` call
   sees the top 10 activating texts and 10 random texts and writes the
   description, with no iteration.
2. **GEPA.** The [`gepa`](https://github.com/gepa-ai/gepa) library. It
   repeatedly revises the description using a `gpt-5.5` reflection step that
   reads the texts the current description misclassified. It sees scores and
   those misclassified texts, and nothing else from the corpus.
3. **Agent.** One OpenAI Agents SDK agent (`gpt-5.5`) per neuron. It has a
   persistent Python sandbox holding the full corpus, all activations, and the
   embeddings, and it can call the official scorer on the evaluation set or on
   any sample of texts it picks.
4. **Agent with held-out scoring.** The same agent, except that its official
   score comes from a held-out set of documents that it can score but never
   read. The held-out section below explains the setup.

GEPA and both agents start from the baseline description, get the same budget
of 1,000 annotator calls per neuron, and must output a single sentence.

## Results on the training evaluation set

| dataset | baseline | GEPA | agent |
|---|---|---|---|
| Yelp (predict the star rating of a review) | 0.755 | 0.888 | **0.962** |
| Congress (predict the speaker's party from a speech) | 0.628 | 0.878 | **0.942** |

Comparing per neuron, the agent beats GEPA on 17 of 20 Yelp neurons and loses
on 2, and it beats GEPA on 14 of 20 Congress neurons and loses on 2. Re-scoring
every winning description with brand new annotations (cache bypassed) moved the
means by at most 0.004, so the scores are not caching artifacts.

**What the sandbox adds.** The agent's corpus access helps most on neurons that
fire on specific words or on several unrelated things at once. For example, one
neuron fires on texts containing the letters "grill", and another fires on
reviews of two specific cheesesteak shops. GEPA makes little progress on those
neurons because reading misclassified texts alone does not reveal the pattern,
while the agent finds the pattern by searching the corpus directly. On neurons
that correspond to one clean concept, GEPA matches the agent and reaches its
score with less of the budget.

**Caveat.** When a neuron does not correspond to a coherent concept, a method
that only chases F1 will "describe" the neuron by listing its evaluation texts.
The single sentence constraint limits that. Where a neuron still cannot be
captured honestly, e.g. Congress neuron 79 at about 0.77, the gap is a sign
that the neuron is not a coherent concept.

## Scores on held-out documents (stages 19 and 20)

The first three methods optimize against an evaluation set they can read, so a
description can score well by fitting those specific texts. To measure how much
of each score depends on fitting the evaluation texts, we built a held-out pool
of documents that no method saw during optimization (2,000 unseen Yelp reviews
and 20,000 unseen Congress chunks) and scored every method's final description
on a held-out evaluation set built the same way as the training one (the pool's
top K activating documents plus K random documents with zero activation).

The fourth method goes further. It is a new run of the agent whose official
score during optimization came from the held-out set. The agent can ask for the
held-out score of any candidate description, but it only receives the overall
precision, recall, and F1. It cannot read the held-out texts or see which ones
it misclassified, so a description can only score well by generalizing.

| held-out F1 (20 neurons) | baseline | GEPA | agent | agent with held-out scoring |
|---|---|---|---|---|
| Yelp | 0.625 | 0.775 | 0.816 | **0.860** |
| Congress | 0.601 | 0.879 | 0.929 | **0.913** |

The agent with held-out scoring beats the original agent's descriptions on Yelp
(0.860 vs 0.816) and roughly matches them on Congress (0.913 vs 0.929). Its
scores on the training set and the held-out set are nearly equal (0.888 and
0.860 on Yelp, 0.913 and 0.913 on Congress), so its descriptions are not fitted
to any particular evaluation set. The cost is a lower score on the readable
training set (0.888 and 0.913, vs 0.962 and 0.942 for the original agent). A
full held-out scoring spends 100 to 200 of the 1,000 annotator calls, and the
agent cannot see which texts it got wrong, so it tries fewer, more distinct
candidates instead of fine-tuning the wording.

## Pipeline (numbered stages, run in order)

The `AGENTIC_DATASET` environment variable selects the dataset (`yelp` by
default, or `congress`). Set `OPENAI_KEY_SAE` for the annotator and the
embeddings, and export `OPENAI_API_KEY` for the agent stages. Shared logic
lives in `harness.py`.

| stage | what it does |
|---|---|
| `01_embed_train_sae.py` | embed the Yelp reviews, train the SAE, save activations |
| `05_congress_prep.py` | the same for a Congress subset of 20,000 documents |
| `02_baseline.py` | write baseline descriptions for the 20 studied neurons |
| `13_rescore_baseline.py` | re-score the baseline on Congress's 200-example protocol |
| `10_agent_v5.py` | run the agent (writes per-neuron records and transcripts) |
| `11_gepa_v5.py` | run GEPA (can be split across processes with `GEPA_SHARD=i/n`) |
| `14_verify.py` | re-score every winning description with the cache bypassed |
| `19_prep_holdout.py` | build the held-out pool (texts and SAE activations) |
| `20_agent_holdout.py` | run the agent with held-out scoring |
| `07_make_report.py` | build `report.html` |
| `16_build_transcript_viewer.py` | build `transcripts.html` |

`watch_report.sh` rebuilds both HTML files as records land during a run.

The artifacts (embeddings, activations, annotation caches, and per-neuron
records) are git-ignored, and the stages above regenerate them. Only the source
files and the two HTML files are tracked.
