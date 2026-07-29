"""Stage 20: blind-holdout agentic optimization.

Same agent, sandbox, budget, and output constraint as stage 10, with one change:
the OFFICIAL metric is computed on a held-out corpus (stage 19) that the agent
can score but never read.

  - Sandbox: the train corpus only (texts/activations/embeddings), exactly as
    in stage 10. No holdout text, activation, or embedding appears in it.
  - score_descriptions(indices=None): scores the held-out official set — the
    holdout pool's top-K activating docs + K seeded zero-activation docs, the
    same protocol shape as stage 10's official set. Returns AGGREGATE
    precision/recall/F1 only: no misclassified indices, nothing readable.
  - score_descriptions(indices=[...]): unchanged — any train-corpus sample,
    full feedback (labels = activation>0, misclassified indices readable).
  - Final pick: argmax held-out official F1 among holdout-scored candidates.
    The record also stores the winner's train-official-set metrics so the
    train->holdout comparison is free at report time.

This tests whether the agent's corpus-probing advantage survives when the
scored texts cannot be memorized: enumeration of readable examples buys
nothing, only properties that generalize to unseen docs score well.
"""
import os, sys, json, io, asyncio, traceback, contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_interp.harness import (load_artifacts, select_neurons, test_indices,
                                    score_description, compute_metrics, ART, N_TEST,
                                    TASK_SPECIFIC_INSTRUCTIONS, DATASET_DESC,
                                    ANNOTATOR_MODEL, MAX_WORDS)
from hypothesaes.annotate import annotate
from hypothesaes.utils import truncate_text

import numpy as np
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, ModelSettings
from openai.types.shared import Reasoning

AGENT_MODEL = "gpt-5.5"
ANNOTATION_BUDGET = 1000
MAX_DESCS_PER_CALL = 5
MAX_TURNS = 60
N_CONCURRENT = 10

RUN_NAME = "agent_runs_holdout"
RUN_DIR = os.path.join(ART, RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)


class FinalResult(BaseModel):
    best_description: str
    rationale: str


INSTRUCTIONS_TEMPLATE = """You are an expert interpretability researcher. A Top-K sparse autoencoder was trained on OpenAI text embeddings of {dataset_desc}. You are studying NEURON {j}. Your goal: find the one-sentence natural-language description of this neuron's firing pattern that maximizes the OFFICIAL fidelity score.

{task_instructions}

OFFICIAL METRIC: an annotator LLM is shown (text, description) pairs and asked yes/no "does this TEXT satisfy this PROPERTY?" on a fixed HELD-OUT evaluation set: the top activating texts + an equal number of fixed random zero-activation texts of a {n_holdout}-document held-out corpus drawn from the same distribution as yours. YOU CANNOT READ THE HELD-OUT CORPUS. Scoring it returns aggregate precision/recall/F1 only — never the texts, never which ones were misclassified. The annotator answers "No" when uncertain and sees ONLY one text + your description, so the description must be a self-contained, objectively checkable property of a single text.

YOUR OBJECTIVE: maximize held-out official F1. Because you cannot see held-out errors, blind hill-climbing with small wording variations wastes budget; instead, learn the neuron's true firing pattern from your readable corpus (its top activating texts mirror the held-out positives' distribution), diagnose candidates on readable samples, and spend held-out scorings on genuinely different candidates to select what generalizes. Prefer the cleanest description that achieves a given score, but coverage beats elegance: if the neuron's concept is genuinely disjunctive, write the disjunction.

TOOLS:
1. run_python(code): persistent Python sandbox (state survives across calls). Preloaded:
   - texts: list of all {n_total} text strings;  stars: np array of the target variable
   - act: np array (n_total,), THIS neuron's activations;  top: np.argsort(-act)
   - all_acts: (n_total, 256) all neuron activations;  emb: (n_total, 1536) unit-norm embeddings
   - np (numpy);  show(i, words=120): truncated text i
   Use print(). Probe hypotheses for free here (keyword counts over positives vs corpus, embedding similarity, co-firing neurons) before spending annotator budget.
2. score_descriptions(descriptions, indices=None): runs the official annotator.
   - indices=None (default): scores on the HELD-OUT official {n_test}-example set — this is the reported metric. Aggregate scores only. Costs {n_test} annotator calls per description.
   - indices=[...]: scores on any text indices you choose from YOUR corpus (labels = activation>0), returning precision/recall/f1 plus the misclassified indices (false_positive_idx/false_negative_idx) — inspect them with show(). Use this for all diagnosis.
   BUDGET: {budget} total annotator calls for this neuron (one call = one text x one description). Repeated (text, description) pairs are cached and FREE. Max {max_per_call} descriptions per call.

REQUIRED PROCESS:
1. Read your corpus's top activating texts across the activation range (top, middle, and just above zero) and zero-activation texts that are semantically close to them.
2. Form hypotheses at multiple specificity levels; keyword-probe them in python first.
3. Your FIRST score_descriptions call on the held-out set must include this previous researcher's description VERBATIM as one candidate (it is your floor):
   "{baseline_desc}"
4. Diagnose weaknesses on readable custom samples (e.g. your corpus's top-K + matched zero-activation texts); read the misclassified texts there, revise, and only then re-score held-out. False positives mean too broad; false negatives mean too narrow or a missing disjunct.
5. Keep iterating while budget remains and you are improving. Stop at F1 = 1.0, or when held-out scores stop improving across genuinely different candidates, or when budget is nearly exhausted (save enough for a final held-out scoring of your best candidate).

STRICT FORMAT REQUIREMENT for every candidate description: a SINGLE sentence (under 40 words), phrased as an objective, self-contained property of one text (like "mentions long wait times to receive service"). It may use a parenthetical e.g.-clause or a genuine disjunction of concrete properties. Do NOT write annotator directives ("answer yes only when...", "Task:", "Answer Yes if..."), multi-step instructions, or bullet lists — the output must read as a description of the neuron's firing pattern, usable as a human-readable hypothesis.

When done, return FinalResult with the best description EXACTLY as scored on the held-out set (verbatim) and a one-paragraph rationale."""


def build_agent_for_neuron(j, texts, stars, acts, emb, hold_texts, hold_acts, state):
    act = acts[:, j]
    h_act = hold_acts[:, j]
    # Same seeded construction as the train official set, applied to the pool.
    h_pos, h_neg = test_indices(hold_acts, j)
    h_pos_locked, h_neg_locked = tuple(h_pos), tuple(h_neg)
    ns = {
        "np": np, "texts": texts, "stars": stars, "act": act,
        "all_acts": acts, "emb": emb, "top": np.argsort(-act),
        "show": lambda i, words=120: truncate_text(texts[int(i)], words),
    }
    holdout_cache = os.path.join(ART, f"annot_cache_holdout_n{j}.json")
    train_cache = os.path.join(ART, f"annot_cache_test_n{j}.json")

    @function_tool
    def run_python(code: str) -> str:
        """Execute Python in this neuron's persistent sandbox. Use print() for output."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)
        except Exception:
            buf.write(traceback.format_exc(limit=3))
        out = buf.getvalue()
        return out[:8000] if out else "(no output — use print())"

    @function_tool
    def score_descriptions(descriptions: list[str], indices: list[int] | None = None) -> str:
        """Score descriptions with the official annotator. indices=None -> held-out official set (reported metric, aggregate scores only); otherwise your chosen train-corpus indices (labels = activation>0, misclassified indices returned)."""
        descriptions = list(dict.fromkeys(d.strip() for d in descriptions if d.strip()))[:MAX_DESCS_PER_CALL]
        official = indices is None
        if official:
            pos_idx, neg_idx = list(h_pos_locked), list(h_neg_locked)
            eval_pool, eval_act, cache = hold_texts, h_act, holdout_cache
        else:
            indices = [int(i) for i in indices][:300]
            pos_idx = [i for i in indices if act[i] > 0]
            neg_idx = [i for i in indices if act[i] == 0]
            if not pos_idx or not neg_idx:
                return json.dumps({"error": "custom sample must contain both activation>0 and activation==0 texts"})
            eval_pool, eval_act, cache = texts, act, train_cache
        n_texts = len(pos_idx) + len(neg_idx)
        cost = n_texts * len(descriptions)
        if state["spent"] + cost > ANNOTATION_BUDGET:
            return json.dumps({"error": f"budget exceeded (spent {state['spent']}/{ANNOTATION_BUDGET}, this call costs {cost}) — return FinalResult"})
        state["spent"] += cost
        idx = list(pos_idx) + list(neg_idx)
        ev_texts = [truncate_text(eval_pool[i], MAX_WORDS) for i in idx]
        labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
        tasks = [(t, d) for d in descriptions for t in ev_texts]
        results = annotate(tasks=tasks, model=ANNOTATOR_MODEL, cache_path=cache,
                           n_workers=40, show_progress=False)
        out = {}
        for d in descriptions:
            ann = np.array([results[d].get(t, 0) for t in ev_texts])
            m = compute_metrics(ann, labels, eval_act[idx])
            m.pop("correlation", None)
            if official:
                # Aggregate feedback only: the held-out set stays unreadable.
                state["scored"][d] = dict(m)
            else:
                m["false_positive_idx"] = [int(idx[i]) for i in range(len(idx))
                                           if ann[i] == 1 and labels[i] == 0][:15]
                m["false_negative_idx"] = [int(idx[i]) for i in range(len(idx))
                                           if ann[i] == 0 and labels[i] == 1][:15]
            out[d] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
        resp = {"results": out,
                "evaluated_on": (f"HELD-OUT official set ({len(pos_idx)}+/{len(neg_idx)}-, unreadable)"
                                 if official else
                                 f"custom train-corpus sample ({n_texts} texts) - NOT the official metric"),
                "annotator_calls_spent": f"{state['spent']}/{ANNOTATION_BUDGET}"}
        if state["scored"]:
            best = max(state["scored"].items(), key=lambda kv: (kv[1]["f1"], kv[1]["precision"]))
            resp["best_official_so_far"] = {"description": best[0],
                                            "f1": round(best[1]["f1"], 3)}
        return json.dumps(resp)

    baselines = json.load(open(os.path.join(ART, "baseline_results.json")))
    baseline_desc = baselines.get(str(j), {}).get("description", "(none available)")
    instructions = INSTRUCTIONS_TEMPLATE.format(
        j=j, task_instructions=TASK_SPECIFIC_INSTRUCTIONS,
        budget=ANNOTATION_BUDGET, max_per_call=MAX_DESCS_PER_CALL,
        n_total=len(texts), n_holdout=len(hold_texts), n_test=N_TEST,
        dataset_desc=DATASET_DESC, baseline_desc=baseline_desc)
    state["instructions"] = instructions
    agent = Agent(name=f"neuron-{j}-holdout", instructions=instructions,
                  tools=[run_python, score_descriptions],
                  model=AGENT_MODEL, output_type=FinalResult,
                  model_settings=ModelSettings(reasoning=Reasoning(summary="auto")))
    return agent, (h_pos, h_neg), holdout_cache


async def optimize_neuron(j, texts, stars, acts, emb, hold_texts, hold_acts, sem):
    async with sem:
        if os.environ.get("SKIP_EXISTING") and os.path.exists(os.path.join(RUN_DIR, f"n{j}.json")):
            rec = json.load(open(os.path.join(RUN_DIR, f"n{j}.json")))
            print(f"[n{j}] skipping (record exists)", flush=True)
            return j, rec
        state = {"spent": 0, "scored": {}}
        agent, (h_pos, h_neg), holdout_cache = build_agent_for_neuron(
            j, texts, stars, acts, emb, hold_texts, hold_acts, state)
        print(f"[n{j}] holdout starting", flush=True)
        claimed, rationale, tool_calls = None, None, []
        for attempt in range(2):
            try:
                result = await Runner.run(agent, input="Begin analyzing the neuron.",
                                          max_turns=MAX_TURNS)
                claimed = result.final_output.best_description.strip()
                rationale = result.final_output.rationale
                try:
                    u = result.context_wrapper.usage
                    state["llm_usage"] = {"requests": u.requests,
                                          "input_tokens": u.input_tokens,
                                          "output_tokens": u.output_tokens}
                except Exception:
                    pass
                transcript = [("SYSTEM PROMPT", state.get("instructions", "")),
                              ("USER", "Begin analyzing the neuron.")]
                for it in result.new_items:
                    t = type(it).__name__
                    if t == "ToolCallItem":
                        ri = it.raw_item
                        tool_calls.append({"tool": getattr(ri, "name", ""),
                                           "args": str(getattr(ri, "arguments", ""))[:3000]})
                        transcript.append(("TOOL CALL: " + getattr(ri, "name", "?"),
                                           str(getattr(ri, "arguments", ""))))
                    elif t == "ToolCallOutputItem":
                        transcript.append(("TOOL OUTPUT", str(getattr(it, "output", ""))[:6000]))
                    elif t == "MessageOutputItem":
                        try:
                            for part in it.raw_item.content:
                                txt = getattr(part, "text", None)
                                if txt:
                                    transcript.append(("ASSISTANT", txt))
                        except Exception:
                            pass
                    elif t == "ReasoningItem":
                        try:
                            for sm in it.raw_item.summary:
                                transcript.append(("REASONING SUMMARY", sm.text))
                        except Exception:
                            pass
                with open(os.path.join(RUN_DIR, f"transcript_n{j}.json"), "w") as jf:
                    json.dump({"transcript": transcript, "claimed": claimed,
                               "rationale": rationale}, jf)
                break
            except Exception as e:
                print(f"[n{j}] agent error (attempt {attempt+1}): {e}", flush=True)
                rationale = f"agent failed: {e}"

        if claimed and claimed not in state["scored"]:
            m = score_description(claimed, h_pos, h_neg, hold_texts, hold_acts[:, j],
                                  cache_path=holdout_cache, return_errors=False)
            state["scored"][claimed] = m
        if not state["scored"]:
            print(f"[n{j}] nothing scored, skipping", flush=True)
            return j, None
        final_desc, final_m = max(state["scored"].items(),
                                  key=lambda kv: (kv[1]["f1"], kv[1]["precision"]))
        # Winner's train-official score, for the train<->holdout comparison.
        t_pos, t_neg = test_indices(acts, j)
        train_m = score_description(final_desc, t_pos, t_neg, texts, acts[:, j],
                                    cache_path=os.path.join(ART, f"annot_cache_test_n{j}.json"),
                                    return_errors=False)
        record = {"neuron": j, "claimed": claimed, "rationale": rationale,
                  "final_description": final_desc,
                  "official_metrics": final_m,
                  "train_official_metrics": train_m,
                  "annotator_calls_spent": state["spent"],
                  "llm_usage": state.get("llm_usage"),
                  "all_scored": state["scored"], "tool_calls": tool_calls}
        with open(os.path.join(RUN_DIR, f"n{j}.json"), "w") as f:
            json.dump(record, f, indent=2)
        print(f"[n{j}] holdout done: holdout_f1={final_m['f1']:.3f} "
              f"train_f1={train_m['f1']:.3f} (spent {state['spent']}) :: {final_desc[:90]}",
              flush=True)
        return j, record


async def main():
    texts, stars, acts, emb = load_artifacts(with_emb=True)
    hold_texts = json.load(open(os.path.join(ART, "holdout_texts.json")))
    hold_acts = np.load(os.path.join(ART, "activations_holdout.npy"))
    neurons = select_neurons(acts, stars)
    only = os.environ.get("AGENT_NEURONS")
    if only:
        keep = {int(x) for x in only.split(",")}
        neurons = [j for j in neurons if j in keep]
    print("blind-holdout optimizing neurons:", neurons, flush=True)
    sem = asyncio.Semaphore(N_CONCURRENT)
    results = await asyncio.gather(*[
        optimize_neuron(j, texts, stars, acts, emb, hold_texts, hold_acts, sem)
        for j in neurons], return_exceptions=True)
    results = [r for r in results if not isinstance(r, BaseException) and r is not None]

    summary = {}
    for j, rec in results:
        if rec is None:
            continue
        summary[str(j)] = {"description": rec["final_description"],
                           "holdout_f1": rec["official_metrics"]["f1"],
                           "train_f1": rec["train_official_metrics"]["f1"],
                           "source": "blind-holdout",
                           "annotator_calls": rec["annotator_calls_spent"]}
        print(f"n{j}: holdout {summary[str(j)]['holdout_f1']:.3f} "
              f"/ train {summary[str(j)]['train_f1']:.3f}", flush=True)
    with open(os.path.join(ART, f"{RUN_NAME}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if summary:
        hs = [v["holdout_f1"] for v in summary.values()]
        ts = [v["train_f1"] for v in summary.values()]
        print(f"BLIND-HOLDOUT AGENT mean F1: holdout {np.mean(hs):.3f}, train {np.mean(ts):.3f}")

if __name__ == "__main__":
    asyncio.run(main())
