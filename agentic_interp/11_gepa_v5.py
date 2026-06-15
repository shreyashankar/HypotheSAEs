"""Stage 11 (v5): vanilla GEPA on the single official set (no dev/test split).

trainset = valset = the official protocol set (100 Yelp / 200 Congress). Vanilla
selection: best_candidate = argmax aggregate val score (seed/baseline is
candidate 0). Budget-matched to the agent: max_metric_calls = 1000 (one metric
call = one text x one description). gpt-5-mini annotates; gpt-5.5 reflects.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_interp.harness import (load_artifacts, select_neurons, test_indices,
                                    score_description, ART, ANNOTATOR_MODEL, MAX_WORDS,
                                    GEPA_RUN_DIRNAME)
from hypothesaes.utils import truncate_text

import gepa
from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from hypothesaes.annotate import annotate

REFLECTION_MODEL = "gpt-5.5"
MAX_METRIC_CALLS = 1000
REFLECTION_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
_reflection_client = None


class DescriptionAdapter(GEPAAdapter):
    """Scores a candidate description against the official set via the repo annotator."""
    def __init__(self, examples, cache_path):
        self.examples = examples  # list of {"text", "label", "act"}
        self.cache_path = cache_path

    def evaluate(self, batch, candidate, capture_traces=False):
        desc = candidate["description"].strip()
        results = annotate(tasks=[(ex["text"], desc) for ex in batch], model=ANNOTATOR_MODEL,
                           cache_path=self.cache_path, n_workers=60, show_progress=False)
        outputs, scores, trajectories = [], [], []
        for ex in batch:
            ann = results.get(desc, {}).get(ex["text"], 0)
            outputs.append(ann)
            scores.append(1.0 if ann == ex["label"] else 0.0)
            trajectories.append({"text": ex["text"], "label": ex["label"], "annotation": ann})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for t in eval_batch.trajectories or []:
            if t["annotation"] == t["label"]:
                continue
            fb = ("This is one of the HIGHEST-activating texts for the neuron, but an annotator "
                  "decided the description does NOT apply to it; it is too narrow or misses a disjunct."
                  if t["label"] == 1 else
                  "This is a random ZERO-activation text, but an annotator decided the description "
                  "DOES apply to it; it is too broad.")
            records.append({"Inputs": {"text": t["text"][:1500]},
                            "Generated Outputs": "annotator said " + ("Yes" if t["annotation"] else "No"),
                            "Feedback": fb})
        if not records:
            records = [{"Inputs": {"text": "(no errors on this minibatch)"},
                        "Generated Outputs": "all correct",
                        "Feedback": "Description is performing well; refine wording for precision."}]
        return {"description": records}


def reflection_lm(prompt: str) -> str:
    global _reflection_client
    if _reflection_client is None:
        from openai import OpenAI
        _reflection_client = OpenAI(api_key=os.environ["OPENAI_KEY_SAE"])
    resp = _reflection_client.responses.create(model=REFLECTION_MODEL, input=prompt)
    REFLECTION_USAGE["calls"] += 1
    REFLECTION_USAGE["input_tokens"] += resp.usage.input_tokens
    REFLECTION_USAGE["output_tokens"] += resp.usage.output_tokens
    return resp.output_text


REFLECTION_TEMPLATE = """A neuron in a neural network fires on certain texts. Its current candidate description is:
```
<curr_param>
```

An annotator LLM was shown (text, description) pairs and asked yes/no "does this TEXT satisfy this PROPERTY?". Below are texts where the annotator's answer disagreed with the neuron's actual behavior, with feedback explaining the direction of the error:
```
<side_info>
```

Write an improved description of the neuron's firing pattern.

STRICT FORMAT REQUIREMENTS: the description must be a SINGLE sentence (under 40 words), phrased as an objective, self-contained property of one text (like "mentions long wait times to receive service"). It may use a parenthetical e.g.-clause with example phrases. Do NOT write multi-step instructions, bullet lists, or directives to the annotator (no "Task:", no "Answer Yes if...").

Provide the new description within ``` blocks."""

GEPA_RUN_DIR = os.path.join(ART, GEPA_RUN_DIRNAME)
os.makedirs(GEPA_RUN_DIR, exist_ok=True)


def main():
    texts, stars, acts, _ = load_artifacts()
    neurons = select_neurons(acts, stars)
    shard = os.environ.get("GEPA_SHARD")
    suffix = ""
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        neurons = neurons[i::n]
        suffix = f"_shard{i}"
    baselines = json.load(open(os.path.join(ART, "baseline_results.json")))
    out = {}
    for j in neurons:
        bdesc = baselines[str(j)]["description"]
        o_pos, o_neg = test_indices(acts, j)
        examples = (
            [{"text": truncate_text(texts[i], MAX_WORDS), "label": 1, "act": float(acts[i, j])} for i in o_pos] +
            [{"text": truncate_text(texts[i], MAX_WORDS), "label": 0, "act": float(acts[i, j])} for i in o_neg])
        cache = os.path.join(ART, f"annot_cache_test_n{j}.json")
        adapter = DescriptionAdapter(examples, cache)

        usage_before = dict(REFLECTION_USAGE)
        print(f"[n{j}] GEPA-v5 starting", flush=True)
        try:
            result = gepa.optimize(
                seed_candidate={"description": bdesc},
                trainset=examples, valset=examples,
                adapter=adapter, reflection_lm=reflection_lm,
                max_metric_calls=MAX_METRIC_CALLS,
                reflection_minibatch_size=5,
                reflection_prompt_template=REFLECTION_TEMPLATE,
                display_progress_bar=False, seed=j,
            )
        except Exception as e:
            print(f"[n{j}] GEPA-v5 failed: {e}", flush=True)
            continue

        best_desc = result.best_candidate["description"].strip()
        m = score_description(best_desc, o_pos, o_neg, texts, acts[:, j], cache_path=cache)
        u = REFLECTION_USAGE
        out[str(j)] = {"description": best_desc, "kept_baseline": best_desc == bdesc,
                       "test_corr": m["correlation"], "test_f1": m["f1"],
                       "n_candidates": len(result.candidates),
                       "metric_calls": result.total_metric_calls,
                       "reflection_usage": {k: u[k] - usage_before[k] for k in u}}
        json.dump(out[str(j)], open(os.path.join(GEPA_RUN_DIR, f"result_n{j}.json"), "w"), indent=2)
        json.dump({"neuron": j, "candidates": [c["description"] for c in result.candidates],
                   "val_scores": list(result.val_aggregate_scores),
                   "discovery_eval_counts": list(result.discovery_eval_counts),
                   "best_idx": int(result.best_idx)},
                  open(os.path.join(GEPA_RUN_DIR, f"n{j}.json"), "w"), indent=2)
        b = baselines[str(j)]["test_metrics"]["correlation"]
        print(f"[n{j}] GEPA-v5 done: {len(result.candidates)} candidates, official {b:.3f} -> "
              f"{m['correlation']:.3f} :: {best_desc[:80]}", flush=True)

    json.dump(out, open(os.path.join(ART, f"gepa_v5_results{suffix}.json"), "w"), indent=2)
    if out:
        cs = [v["test_corr"] for v in out.values()]
        print(f"GEPA-v5 shard mean official correlation: {np.mean(cs):.3f}")


if __name__ == "__main__":
    main()
