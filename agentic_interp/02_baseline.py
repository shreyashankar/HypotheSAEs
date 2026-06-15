"""Stage 2: baseline descriptions via the repo's default pipeline, scored on TEST."""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentic_interp.harness import (load_artifacts, select_neurons, test_indices,
                                    score_description, TASK_SPECIFIC_INSTRUCTIONS, ART)
from hypothesaes.interpret_neurons import NeuronInterpreter, InterpretConfig

INTERPRETER_MODEL = "gpt-5.2"

texts, stars, acts, _ = load_artifacts()
neurons = select_neurons(acts, stars)
print("Selected neurons:", neurons)

interpreter = NeuronInterpreter(interpreter_model=INTERPRETER_MODEL,
                                n_workers_interpretation=10)
config = InterpretConfig(task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS)
interps = interpreter.interpret_neurons(texts=texts, activations=acts,
                                        neuron_indices=neurons, config=config)

results = {}
for j in neurons:
    desc = interps[j][0]
    print(f"\nneuron {j}: {desc}")
    t_pos, t_neg = test_indices(acts, j)
    cache = os.path.join(ART, f"annot_cache_test_n{j}.json")
    metrics = score_description(desc, t_pos, t_neg, texts, acts[:, j], cache_path=cache)
    print("  test:", {k: round(v, 3) for k, v in metrics.items()})
    results[j] = {"description": desc, "test_metrics": metrics}

with open(os.path.join(ART, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)

corrs = [r["test_metrics"]["correlation"] for r in results.values()]
f1s = [r["test_metrics"]["f1"] for r in results.values()]
print(f"\nBASELINE mean test correlation: {np.mean(corrs):.3f}, mean F1: {np.mean(f1s):.3f}")
