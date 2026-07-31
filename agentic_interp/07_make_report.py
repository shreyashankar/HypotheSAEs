"""Generate report.html — three arms (baseline / GEPA / agent), live-updating.

Reads per-neuron run records as they land, so it can be re-run at any time
during the experiment to get a current snapshot.
"""
import os, sys, json, glob, io, base64, html
from datetime import datetime, timezone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from hypothesaes.utils import truncate_text
from hypothesaes.annotate import generate_cache_key
# test_indices/compute_metrics shared so the report's eval set + F1 can never
# drift from what 02/10/11/14 actually scored. Pass n_test explicitly since this
# process handles both datasets. AGENT/GEPA dir names shared with the writers.
from agentic_interp.harness import (test_indices, compute_metrics,
                                    AGENT_RUN_DIRNAME, GEPA_RUN_DIRNAME)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS = [
    ("Yelp", os.path.join(BASE, "artifacts"), 100, """
<p><b>Prediction task.</b> The HypotheSAEs use case here: explain <b>what features of a Yelp restaurant review
predict its 1–5 star rating</b>. Data: 10,000 reviews from the repo's demo subset of the paper's Yelp dataset
(one text = one full review, avg ~100 words). Each review is embedded with text-embedding-3-small; the SAE learns
256 sparse features over those embeddings; descriptions of the predictive features become the human-readable
hypotheses (e.g., "mentions long wait times" &rarr; lower rating). The 20 neurons studied = the 10 whose
activations correlate most strongly (|r|) with star rating + 10 random alive neurons. Official evaluation set per
neuron: its top-50 activating reviews + 50 fixed random zero-activation reviews.</p>"""),
    ("Congress", os.path.join(BASE, "artifacts_congress"), 200, """
<p><b>Prediction task.</b> Explain <b>what features of a U.S. Congressional speech predict the speaker's party</b>
(republican = 1, democrat = 0). Data: 20,000 ten-sentence speech chunks sampled from the paper's Congress dataset
(114K chunks, sourced from the Stanford congress_text corpus). Same pipeline: embeddings &rarr; 256-neuron SAE
&rarr; describe the neurons whose activations are most party-predictive (10 by |r|) + 10 random alive neurons.
Official evaluation set per neuron: its top-100 activating chunks + 100 fixed random zero-activation chunks
(the paper reproduction protocol, n=200).</p>"""),
]

COLORS = {"baseline": "#888888", "gepa": "#8e44ad", "agent": "#2e7d32"}
ANNOT_COST = 0.000556  # measured: 572 in / 206 out tokens on gpt-5-mini
G55_IN, G55_OUT = 5.0 / 1e6, 30.0 / 1e6  # gpt-5.5 standard pricing ($5/M in, $30/M out)


def method_cost(d, mkey, n_test):
    """Return (cost_string, dollars) for one neuron+method."""
    if mkey == "baseline":
        return f"~${n_test * ANNOT_COST + 0.01:.2f}", n_test * ANNOT_COST + 0.01
    if mkey == "agent":
        ann = d[mkey].get("spent", 0) * ANNOT_COST
        u = d[mkey].get("llm_usage")
        if u:
            llm = u["input_tokens"] * G55_IN + u["output_tokens"] * G55_OUT
            return f"${ann + llm:.2f} (annot ${ann:.2f} + agent ${llm:.2f})", ann + llm
        return f"~${ann:.2f}+ (annot only; agent tokens not recorded)", ann
    if mkey == "gepa":
        mc = d[mkey].get("metric_calls")
        ru = d[mkey].get("reflection_usage")
        if mc is not None and ru:
            ann = mc * ANNOT_COST
            llm = ru["input_tokens"] * G55_IN + ru["output_tokens"] * G55_OUT
            return f"${ann + llm:.2f} (annot ${ann:.2f} + reflect ${llm:.2f})", ann + llm
        return "n/a", 0.0
    return "n/a", 0.0


def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def esc(s):
    return html.escape(str(s))


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def collect(art, n_test):
    """Gather per-neuron data for all three arms; tolerant of partial results."""
    acts = np.load(os.path.join(art, "activations_train.npy"))
    baseline = load(os.path.join(art, "baseline_results.json")) or {}
    data = {}
    official = {}  # j -> (tp, tn), computed once and reused by the GEPA loop
    for j_str, rec in baseline.items():
        j = int(j_str)
        tp, tn = test_indices(acts, j, n_test)
        official[j] = (tp, tn)
        labels = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
        a = acts[list(tp) + list(tn), j]
        ceiling = float(np.corrcoef(a, labels)[0, 1])
        n_pos_corpus = int((acts[:, j] > 0).sum())
        data[j] = {"ceiling": ceiling,
                   "n_pos_corpus": n_pos_corpus,
                   "n_off_pos": len(tp), "n_off_neg": len(tn),
                   "n_midlow": n_pos_corpus - len(tp),
                   "baseline": {"corr": rec["test_metrics"]["correlation"],
                                "f1": rec["test_metrics"]["f1"],
                                "desc": rec["description"]}}
    agent_dir = AGENT_RUN_DIRNAME

    def agent_budget_traj(j):
        tpath = os.path.join(art, agent_dir, f"transcript_n{j}.json")
        if not os.path.exists(tpath):
            return None
        rec = load(tpath)
        if not rec:
            return None
        pts, pending_official = [], False
        for role, txt in rec["transcript"]:
            if role.startswith("TOOL CALL") and "score_descriptions" in role:
                try:
                    args = json.loads(txt)
                    pending_official = not args.get("indices")
                except Exception:
                    pending_official = False
            elif role == "TOOL OUTPUT":
                try:
                    obj = json.loads(txt)
                except Exception:
                    continue
                if not isinstance(obj, dict) or "results" not in obj or "annotator_calls_spent" not in obj:
                    continue
                spent = int(str(obj["annotator_calls_spent"]).split("/")[0])
                ev = str(obj.get("evaluated_on", ""))
                if pending_official or ev.startswith("FULL"):
                    f1s = [m.get("f1") for m in obj["results"].values()
                           if isinstance(m, dict) and m.get("f1") is not None]
                    if f1s:
                        pts.append((spent, max(f1s)))
        return pts or None

    for f in glob.glob(os.path.join(art, agent_dir, "n*.json")):
        rec = load(f)
        if not rec:
            continue
        j = rec["neuron"]
        if j in data:
            data[j]["agent"] = {"corr": rec["official_metrics"].get("correlation", 0.0),
                                "f1": rec["official_metrics"]["f1"],
                                "desc": rec["final_description"],
                                "all_scored": rec.get("all_scored", {}),
                                "rationale": rec.get("rationale", ""),
                                "tool_calls": rec.get("tool_calls", []),
                                "spent": rec.get("annotator_calls_spent", 0),
                                "llm_usage": rec.get("llm_usage")}
            data[j]["agent"]["budget_traj"] = agent_budget_traj(j)
            if os.path.exists(os.path.join(art, agent_dir, f"transcript_n{j}.json")):
                dskey = "yelp" if os.path.basename(art) == "artifacts" else "congress"
                data[j]["agent"]["transcript"] = f"transcripts.html#{dskey}_agent-n{j}"
    texts_recs = load(os.path.join(art, "train_texts.json")) or []
    all_texts = [r["text"] for r in texts_recs]
    for f in glob.glob(os.path.join(art, GEPA_RUN_DIRNAME, "n*.json")):
        rec = load(f)
        if not rec:
            continue
        j = rec["neuron"]
        if j not in data or not all_texts:
            continue
        cache = load(os.path.join(art, f"annot_cache_test_n{j}.json")) or {}
        tp, tn = official[j]
        idx = list(tp) + list(tn)
        ev = [truncate_text(all_texts[i], 256) for i in idx]
        labels = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
        activ = acts[idx, j]
        traj = []
        for desc in rec.get("candidates", []):
            try:
                ann = np.array([cache[generate_cache_key(desc, t)] for t in ev])
            except KeyError:
                continue
            traj.append(compute_metrics(ann, labels, activ)["f1"])
        if traj:
            counts = rec.get("discovery_eval_counts", [])
            data[j].setdefault("gepa_traj",
                               list(zip(counts, traj)) if len(counts) == len(traj)
                               else [(i * n_test, v) for i, v in enumerate(traj)])
    for f in glob.glob(os.path.join(art, GEPA_RUN_DIRNAME, "result_n*.json")):
        rec = load(f)
        if not rec:
            continue
        j = int(os.path.basename(f)[len("result_n"):-len(".json")])
        if j in data:
            dskey = "yelp" if os.path.basename(art) == "artifacts" else "congress"
            data[j]["gepa"] = {"transcript": f"transcripts.html#{dskey}_gepa-n{j}",
                               "corr": rec.get("test_corr", 0.0), "f1": rec["test_f1"],
                               "desc": rec["description"],
                               "n_candidates": rec.get("n_candidates", 0),
                               "metric_calls": rec.get("metric_calls"),
                               "reflection_usage": rec.get("reflection_usage")}
    return data


def bars_fig(data, title):
    methods = [("baseline", "Baseline (repo one-shot)"),
               ("gepa", "GEPA (reflective evolution)"),
               ("agent", "Agent (sandbox + scorer)")]
    fig, axes = plt.subplots(1, 1, figsize=(6.5, 2.6))
    for ax, key, ttl in [(axes, "f1", "Mean official F1 (1.0 = perfect discrimination)")]:
        labels, vals, cols = [], [], []
        for mkey, mlabel in methods:
            done = [d[mkey][key] for d in data.values() if mkey in d]
            if done:
                labels.append(f"{mlabel}  (n={len(done)})")
                vals.append(np.mean(done))
                cols.append(COLORS[mkey])
        bars = ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlim(0, 1.02); ax.set_title(ttl, fontsize=11)
        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return b64fig(fig)


def trajectory_fig(data, title):
    """Agent hill-climb: best-so-far official corr per officially-scored candidate."""
    neurons = sorted(j for j, d in data.items() if "agent" in d and d["agent"]["all_scored"])
    if not neurons:
        return None
    ncol = 5
    nrow = int(np.ceil(len(neurons) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow), sharey=True)
    axes = np.atleast_2d(axes)
    for k, j in enumerate(neurons):
        ax = axes[k // ncol][k % ncol]
        d = data[j]
        ax.axhline(1.0, color="#d4ac0d", ls=":", lw=1.4, label="perfect (F1=1)")
        ax.axhline(d["baseline"]["f1"], color="#888", ls="--", lw=1.2, label="baseline")
        if "gepa_traj" in d:
            xs = [p[0] for p in d["gepa_traj"]]
            ys = list(np.maximum.accumulate([p[1] for p in d["gepa_traj"]]))
            ax.step(xs + [1000], ys + [ys[-1]], where="post", color=COLORS["gepa"], lw=1.6,
                    ls="--", label="GEPA best-so-far")
        elif "gepa" in d:
            ax.axhline(d["gepa"]["f1"], color=COLORS["gepa"], ls="-.", lw=1.2, label="GEPA final")
        bt = d["agent"].get("budget_traj")
        if bt:
            xs = [p[0] for p in bt]
            ys = list(np.maximum.accumulate([p[1] for p in bt]))
            ax.step(xs + [1000], ys + [ys[-1]], where="post",
                    color=COLORS["agent"], lw=1.8, label="agent best-so-far")
            ax.plot(xs[0], ys[0], "o", color=COLORS["agent"], ms=4)
        else:
            f1s = [m["f1"] for m in d["agent"]["all_scored"].values()]
            best = np.maximum.accumulate(f1s)
            ax.step(range(len(best)), best, where="post", color=COLORS["agent"], lw=1.8, label="agent best-so-far")
        ax.set_xlim(0, 1000)
        ax.set_title(f"neuron {j}", fontsize=9)
        if k >= len(neurons) - ncol:
            ax.set_xlabel("annotator calls spent", fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for k in range(len(neurons), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"{title} — optimization progress: best F1 so far vs. annotator calls spent (budget = 1,000)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return b64fig(fig)


def cost_accuracy_svg(data, n_test, title, dskey=''):
    """Interactive SVG scatter: x = official F1, y = cost ($, log), hover = neuron."""
    import math
    W, H = 860, 420
    L, R, T, B = 70, 175, 40, 50
    pts = []
    for j, d in data.items():
        for mkey, label in [("baseline", "baseline"), ("gepa", "GEPA"), ("agent", "agent")]:
            if mkey not in d:
                continue
            _, cost = method_cost(d, mkey, n_test)
            if cost <= 0:
                continue
            pts.append((d[mkey]["f1"], cost, mkey, j, d["n_pos_corpus"]))
    if not pts:
        return ""
    fmin = max(0.0, min(p[0] for p in pts) - 0.05)
    cmin = min(p[1] for p in pts) * 0.7
    cmax = max(p[1] for p in pts) * 1.4
    def X(c):
        return L + (math.log10(cmax) - math.log10(c)) / (math.log10(cmax) - math.log10(cmin)) * (W - L - R)
    def Y(f1):
        return T + (1 - (f1 - fmin) / (1.001 - fmin)) * (H - T - B)
    out = [f"<svg viewBox='0 0 {W} {H}' style='max-width:100%;background:#fff;border:1px solid #eee;border-radius:8px'>"]
    out.append(f"<text x='{W/2}' y='20' text-anchor='middle' font-size='14' font-weight='600'>{title}</text>")
    # x ticks: cost (log)
    candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    for c in candidates:
        if cmin <= c <= cmax:
            cx = X(c)
            out.append(f"<line x1='{cx:.0f}' y1='{T}' x2='{cx:.0f}' y2='{H-B}' stroke='#f0f0f0'/>")
            out.append(f"<text x='{cx:.0f}' y='{H-B+16}' text-anchor='middle' font-size='10' fill='#666'>${c:g}</text>")
    out.append(f"<text x='{(L+W-R)/2}' y='{H-B+34}' text-anchor='middle' font-size='11' fill='#444'>cost per neuron (USD, log scale; cheaper &rarr;)</text>")
    # y ticks: F1
    f = fmin
    while f <= 1.001:
        fy = Y(f)
        out.append(f"<line x1='{L}' y1='{fy:.0f}' x2='{W-R}' y2='{fy:.0f}' stroke='#f0f0f0'/>")
        out.append(f"<text x='{L-6}' y='{fy:.0f}' text-anchor='end' font-size='10' fill='#666'>{f:.2f}</text>")
        f = round(f + max(0.05, round((1 - fmin) / 8, 2)), 2)
    out.append(f"<text x='16' y='{(T+H-B)/2}' font-size='11' fill='#444' transform='rotate(-90 16 {(T+H-B)/2})' text-anchor='middle'>official-set F1</text>")
    out.append(f"<line x1='{L}' y1='{H-B}' x2='{W-R}' y2='{H-B}' stroke='#999'/>")
    out.append(f"<line x1='{L}' y1='{T}' x2='{L}' y2='{H-B}' stroke='#999'/>")
    import math as _m
    npos_vals = [p[4] for p in pts]
    lo, hi = _m.log10(min(npos_vals)), _m.log10(max(npos_vals) + 1)
    def RAD(npos):
        if hi <= lo:
            return 6.0
        return 3.5 + 8.5 * (_m.log10(npos) - lo) / (hi - lo)
    for f1, cost, mkey, j, npos in sorted(pts, key=lambda t: -t[4]):
        r = RAD(npos)
        href = ""
        if mkey == "agent":
            href = f"transcripts.html#{dskey}_agent-n{j}"
        elif mkey == "gepa":
            href = f"transcripts.html#{dskey}_gepa-n{j}"
        out.append(f"<circle cx='{X(cost):.1f}' cy='{Y(f1):.1f}' r='{r:.1f}' fill='{COLORS[mkey]}' fill-opacity='0.8' stroke='#fff' stroke-width='1' "
                   f"class='pt' data-n='{j}' data-r='{r:.1f}' data-href='{href}' "
                   f"data-tip='neuron {j} — {mkey}: F1 {f1:.3f}, ${cost:.2f}, fires on {npos} texts'/>")
    # legend: vertical block to the right of the plot
    lx, ly = W - R + 22, T + 14
    for mkey, lbl in [("baseline", "baseline"), ("gepa", "GEPA"), ("agent", "agent")]:
        out.append(f"<circle cx='{lx}' cy='{ly}' r='6' fill='{COLORS[mkey]}'/>")
        out.append(f"<text x='{lx+12}' y='{ly+4}' font-size='11'>{lbl}</text>")
        ly += 22
    ly += 8
    for line in ["dot size =", "# texts the neuron", "fires on (log)"]:
        out.append(f"<text x='{lx-6}' y='{ly}' font-size='10' fill='#777'>{line}</text>")
        ly += 13
    out.append("</svg>")
    return "".join(out)


def neuron_table(data, table_id, n_test):
    """Interactive: click a row to expand all three methods' full descriptions."""
    def fmt(d, key):
        if key not in d:
            return "…"
        return f"{d[key]['f1']:.3f}"

    rows = ""
    order = sorted(data.keys(),
                   key=lambda j: -(data[j].get("agent", {}).get("f1", -1) - data[j]["baseline"]["f1"]))
    for j in order:
        d = data[j]
        if "agent" in d:
            cls = ("win" if d["agent"]["f1"] > d["baseline"]["f1"] + 1e-9
                   else ("tie" if abs(d["agent"]["f1"] - d["baseline"]["f1"]) <= 1e-9 else "loss"))
        else:
            cls = "tie"
        rid = f"{table_id}_n{j}"
        a_f1 = d.get("agent", {}).get("f1")
        g_f1 = d.get("gepa", {}).get("f1")
        d_ab = (a_f1 - d["baseline"]["f1"]) if a_f1 is not None else -99
        d_ag = (a_f1 - g_f1) if (a_f1 is not None and g_f1 is not None) else -99
        rows += (f"<tr class='{cls} clickable mainrow' data-rid='{rid}' "
                 f"data-ab='{d_ab:.4f}' data-ag='{d_ag:.4f}' data-af='{a_f1 if a_f1 is not None else -99:.4f}' "
                 f"onclick=\"toggleRow('{rid}')\">"
                 f"<td>n{j} &#9656;</td>"
                 f"<td>{d['n_pos_corpus']}</td>"
                 f"<td>{d['n_off_pos']}+ / {d['n_off_neg']}&minus;</td>"
                 f"<td>{d['n_midlow']}</td>"
                 f"<td>{fmt(d, 'baseline')}</td><td>{fmt(d, 'gepa')}</td><td>{fmt(d, 'agent')}</td></tr>")
        detail = "<table class='inner'>"
        for mkey, mlabel in [("baseline", "Baseline"), ("gepa", "GEPA"), ("agent", "Agent")]:
            if mkey in d:
                m = d[mkey]
                link = (f" · <a href='{m['transcript']}' target='_blank'>full transcript</a>"
                        if m.get("transcript") else "")
                cost_str, _ = method_cost(d, mkey, n_test)
                detail += (f"<tr><td class='mname'>{mlabel}<br><span class='muted'>F1 {m['f1']:.3f}"
                           f"{link}<br>cost {cost_str}</span></td><td class='desc'>{esc(m['desc'])}</td></tr>")
            else:
                detail += f"<tr><td class='mname'>{mlabel}</td><td class='muted'>(still running)</td></tr>"
        detail += "</table>"
        rows += f"<tr id='{rid}' class='detail' style='display:none'><td colspan='7'>{detail}</td></tr>"
    header = ("<tr><th>neuron</th><th title='texts with activation>0 in the whole corpus'># act&gt;0<br>(corpus)</th>"
              "<th title='official evaluation set composition'>official set</th>"
              "<th title='firing texts excluded from evaluation (below the top-N cutoff)'># firing texts<br>outside official set</th>"
              "<th>baseline F1</th><th>GEPA F1</th><th>agent F1</th></tr>")
    controls = (f"<p class='muted'>Click any row to see all three methods' descriptions and the agent transcript. "
                f"F1 = 1.0 means perfect discrimination on the official set. Sort by: "
                f"<select id='sk_{table_id}' onchange=\"resort('{table_id}')\">"
                f"<option value='ab'>agent − baseline</option>"
                f"<option value='ag'>agent − GEPA</option>"
                f"<option value='af'>agent F1</option></select> "
                f"<select id='sd_{table_id}' onchange=\"resort('{table_id}')\">"
                f"<option value='desc'>descending</option>"
                f"<option value='asc'>ascending</option></select></p>")
    return controls + f"<table class='cands' id='tbl_{table_id}'>{header}{rows}</table>"


def blind_holdout_section(art, n_test, data):
    """Stage 19/20 results: the agent re-run with its official metric moved to a
    held-out corpus it can score but never read. Frozen-arm comparisons on the
    same held-out sets come entirely from the annotation caches (no API calls)."""
    recs = {}
    for f in glob.glob(os.path.join(art, "agent_runs_holdout", "n*.json")):
        r = load(f)
        if r and "official_metrics" in r:
            recs[r["neuron"]] = r
    h_texts = load(os.path.join(art, "holdout_texts.json"))
    a_path = os.path.join(art, "activations_holdout.npy")
    if not recs or not h_texts or not os.path.exists(a_path):
        return ""
    hold_acts = np.load(a_path)

    def frozen_holdout_f1(j, desc):
        cache = load(os.path.join(art, f"annot_cache_holdout_n{j}.json")) or {}
        tp, tn = test_indices(hold_acts, j, n_test)
        idx = list(tp) + list(tn)
        ev = [truncate_text(h_texts[i], 256) for i in idx]
        try:
            ann = np.array([cache[generate_cache_key(desc, t)] for t in ev])
        except KeyError:
            return None
        labels = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
        return compute_metrics(ann, labels, hold_acts[idx, j])["f1"]

    rows_per_neuron, sums = {}, {k: [] for k in ("baseline", "gepa", "agent",
                                                 "blind_h", "blind_t", "frozen_train")}
    for j in sorted(recs):
        if j not in data:
            continue
        r = recs[j]
        row = {"blind_h": r["official_metrics"]["f1"],
               "blind_t": r.get("train_official_metrics", {}).get("f1"),
               "blind_desc": r["final_description"], "spent": r.get("annotator_calls_spent")}
        for mkey in ("baseline", "gepa", "agent"):
            row[mkey] = (frozen_holdout_f1(j, data[j][mkey]["desc"])
                         if mkey in data[j] else None)
        row["agent_desc"] = data[j].get("agent", {}).get("desc", "")
        rows_per_neuron[j] = row
        if all(row[k] is not None for k in ("baseline", "gepa", "agent", "blind_t")):
            for k in ("baseline", "gepa", "agent", "blind_h", "blind_t"):
                sums[k].append(row[k])
            sums["frozen_train"].append(data[j]["agent"]["f1"])
    n = len(sums["blind_h"])
    if not n:
        return ""

    def td(v):
        return f"<td>{v:.3f}</td>" if v is not None else "<td class='muted'>–</td>"

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    labels = [("baseline", "Baseline"), ("gepa", "GEPA"),
              ("agent", "Agent"), ("blind_h", "Agent with held-out scoring")]
    vals = [np.mean(sums[k]) for k, _ in labels]
    cols = [COLORS["baseline"], COLORS["gepa"], COLORS["agent"], "#1565c0"]
    bars = ax.barh(range(4), vals, color=cols)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{lbl}  (n={n})" for _, lbl in labels], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_title("Mean F1 on held-out documents", fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    holdout_bars = b64fig(fig)

    arm_rows = (
        f"<tr><td class='mname'>Baseline</td><td>one LLM call over training examples</td>"
        f"<td>{np.mean([data[j]['baseline']['f1'] for j in rows_per_neuron if j in data]):.3f}</td>"
        f"<td>{np.mean(sums['baseline']):.3f}</td></tr>"
        f"<tr><td class='mname'>GEPA</td><td>iterated on the training evaluation set and read its "
        f"misclassified texts</td>"
        f"<td>{np.mean([data[j]['gepa']['f1'] for j in rows_per_neuron if 'gepa' in data.get(j, {})]):.3f}</td>"
        f"<td>{np.mean(sums['gepa']):.3f}</td></tr>"
        f"<tr><td class='mname'>Agent</td><td>iterated on the training evaluation set and could read "
        f"the whole training corpus</td>"
        f"<td>{np.mean(sums['frozen_train']):.3f}</td><td>{np.mean(sums['agent']):.3f}</td></tr>"
        f"<tr style='background:#e8f0fe'><td class='mname'>Agent with held-out scoring</td>"
        f"<td>iterated on the held-out score alone and could never read the held-out documents</td>"
        f"<td>{np.mean(sums['blind_t']):.3f}</td><td><b>{np.mean(sums['blind_h']):.3f}</b></td></tr>")
    detail = ""
    for j, row in rows_per_neuron.items():
        detail += (f"<tr><td>n{j}</td>{td(row['baseline'])}{td(row['gepa'])}"
                   f"{td(row['agent'])}{td(row['blind_h'])}</tr>"
                   f"<tr class='detail'><td colspan='5'><span class='muted'>agent:</span> "
                   f"{esc(row['agent_desc'])}<br><span class='muted'>agent with held-out scoring:</span> "
                   f"{esc(row['blind_desc'])}</td></tr>")
    n_per_class = n_test // 2
    return (
        "<h3 style='background:#e8f0fe;padding:6px 10px;border-radius:6px'>Scores on held-out documents</h3>"
        f"<img src='data:image/png;base64,{holdout_bars}'/>"
        f"<p class='muted'>Everything else on this page is scored on an evaluation set drawn from the training "
        f"corpus, which the methods could read while optimizing. The table below instead scores every method on "
        f"documents that no method ever saw. The held-out evaluation set for each neuron contains the "
        f"{n_per_class} held-out documents where the neuron activates most strongly, plus {n_per_class} random "
        f"held-out documents where it does not activate at all, drawn from a pool of {len(h_texts):,} documents "
        f"outside the training corpus.</p>"
        f"<p class='muted'>The first three rows take the descriptions reported above and score them again on the "
        f"held-out set. The last row is a new run of the agent whose official score came from the held-out set "
        f"during optimization. That agent could ask for the held-out score of any candidate description, but it "
        f"only ever received the overall precision, recall, and F1. It could not read the held-out texts or see "
        f"which ones were misclassified, so a description could only score well by generalizing. Its sandbox, "
        f"budget (1,000 annotator calls), and output constraints are unchanged. n={n} neurons.</p>"
        "<table class='cands'><tr><th>method</th><th>how it was optimized</th>"
        "<th>F1 on the training set</th><th>F1 on the held-out set</th></tr>"
        f"{arm_rows}</table>"
        "<details><summary>Per-neuron held-out F1 and each agent's description</summary>"
        "<table class='cands'><tr><th>neuron</th><th>baseline</th><th>GEPA</th>"
        "<th>agent</th><th>agent with held-out scoring</th></tr>"
        f"{detail}</table></details>")


def evolution_blocks(data, top_k=2):
    done = [(j, d) for j, d in data.items() if "agent" in d and d["agent"]["all_scored"]]
    done.sort(key=lambda jd: -(jd[1]["agent"]["f1"] - jd[1]["baseline"]["f1"]))
    out = ""
    for j, d in done[:top_k]:
        out += (f"<h4>neuron {j} — baseline F1 {d['baseline']['f1']:.3f} → agent "
                f"F1 {d['agent']['f1']:.3f}</h4>")
        out += f"<p><i>{esc(d['agent']['rationale'][:500])}</i></p>"
        rows = "".join(f"<tr><td>{i}</td><td>{m['f1']:.2f}</td>"
                       f"<td class='desc'>{esc(desc)}</td></tr>"
                       for i, (desc, m) in enumerate(d["agent"]["all_scored"].items()))
        out += ("<details><summary>All officially-scored candidates, in order</summary>"
                "<table class='cands'><tr><th>#</th><th>F1</th><th>description</th></tr>"
                + rows + "</table></details>")
    return out


HEAD = """<!DOCTYPE html><html><head><meta charset='utf-8'><link rel='preconnect' href='https://fonts.googleapis.com'><link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' rel='stylesheet'><title>Hill-climbing SAE feature descriptions</title>
<style>
body { font-family: 'Inter', -apple-system, Helvetica, Arial, sans-serif; max-width: 1020px; margin: 2em auto; color: #222; line-height: 1.45; padding: 0 1em; }
h1 { font-size: 1.55em; } h2 { margin-top: 1.6em; border-bottom: 2px solid #eee; padding-bottom: 4px; }
img { max-width: 100%; margin: 0.8em 0; }
table.cands { border-collapse: collapse; font-size: 0.82em; margin: 0.6em 0; width: 100%; }
table.cands th, table.cands td { border: 1px solid #ddd; padding: 3px 7px; text-align: left; vertical-align: top; }
td.desc { max-width: 600px; }
tr.win td { background: #eaf5ea; } tr.loss td { background: #fdecec; } tr.tie td { background: #fafafa; }
pre.code { background: #f6f8fa; padding: 0.7em; font-size: 0.78em; overflow-x: auto; border-radius: 6px; }
p.toolhead { font-weight: 600; margin: 0.7em 0 0.2em; color: #555; font-size: 0.85em; }
ul.descs { font-size: 0.85em; background: #f0f4fa; padding: 0.6em 1.6em; border-radius: 6px; }
.muted { color: #999; } details { margin: 0.6em 0; }
.statusbar { background: #fff8e6; border: 1px solid #f0e0b0; border-radius: 6px; padding: 0.5em 0.9em; font-size: 0.9em; }
tr.clickable { cursor: pointer; }
tr.clickable:hover td { filter: brightness(0.95); }
tr.detail > td { background: #fcfcfd; border-top: none; }
table.inner { width: 100%; border-collapse: collapse; }
table.inner td { border: none; border-bottom: 1px solid #eee; padding: 5px 8px; font-size: 0.95em; }
td.mname { width: 110px; font-weight: 600; vertical-align: top; }
</style>
<script>
function toggleRow(id) {
  var r = document.getElementById(id);
  r.style.display = (r.style.display === 'none') ? 'table-row' : 'none';
}
function resort(tid) {
  var key = document.getElementById('sk_' + tid).value;
  var dir = document.getElementById('sd_' + tid).value === 'desc' ? -1 : 1;
  var tbl = document.getElementById('tbl_' + tid);
  var pairs = [];
  tbl.querySelectorAll('tr.mainrow').forEach(function(r) {
    var detail = document.getElementById(r.getAttribute('data-rid'));
    pairs.push([parseFloat(r.getAttribute('data-' + key)), r, detail]);
  });
  pairs.sort(function(a, b) { return dir * (a[0] - b[0]); });
  pairs.forEach(function(pr) { tbl.appendChild(pr[1]); if (pr[2]) tbl.appendChild(pr[2]); });
}
document.addEventListener('DOMContentLoaded', function() {
  var tip = document.createElement('div');
  tip.style.cssText = 'position:fixed;display:none;background:#222;color:#fff;padding:5px 10px;border-radius:5px;font-size:12px;pointer-events:auto;z-index:99;white-space:nowrap';
  document.body.appendChild(tip);
  var hideTimer = null;
  function scheduleHide(c) {
    hideTimer = setTimeout(function() { tip.style.display = 'none'; if (c) c.setAttribute('r', c.getAttribute('data-r')); }, 350);
  }
  tip.addEventListener('mouseenter', function() { clearTimeout(hideTimer); });
  tip.addEventListener('mouseleave', function() { scheduleHide(null); });
  document.querySelectorAll('circle.pt').forEach(function(c) {
    c.addEventListener('mousemove', function(e) {
      clearTimeout(hideTimer);
      var href = c.getAttribute('data-href');
      tip.innerHTML = c.getAttribute('data-tip') +
        (href ? " &middot; <a href='" + href + "' target='_blank' style='color:#9ecbff'>transcript</a>" : "");
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY - 10) + 'px';
      tip.style.display = 'block';
      c.setAttribute('r', String(parseFloat(c.getAttribute('data-r')) + 2));
    });
    c.addEventListener('mouseleave', function() { scheduleHide(c); });
    c.addEventListener('click', function(e) {
      e.stopPropagation();
      var svg = c.ownerSVGElement;
      var n = c.getAttribute('data-n');
      var already = svg.getAttribute('data-focus') === n;
      svg.querySelectorAll('circle.pt').forEach(function(o) {
        o.setAttribute('fill-opacity', (already || o.getAttribute('data-n') === n) ? '0.8' : '0.12');
      });
      svg.setAttribute('data-focus', already ? '' : n);
    });
  });
  document.addEventListener('click', function() {
    document.querySelectorAll('svg[data-focus]').forEach(function(svg) {
      svg.querySelectorAll('circle.pt').forEach(function(o) { o.setAttribute('fill-opacity', '0.8'); });
      svg.setAttribute('data-focus', '');
    });
  });
});
</script>
</head><body>
"""

INTRO = """
<h1>Hill-climbing SAE feature descriptions (HypotheSAEs)</h1>
<p><b>Task.</b> For each SAE neuron, find the one-sentence description that best discriminates its top-activating
texts from random non-activating texts, as judged by the repo's official protocol: gpt-5-mini answers yes/no
"does this TEXT satisfy this PROPERTY?" independently per text, and the score is the <b>F1</b> of those answers
against the labels (top-activating = positive, zero-activation = negative). F1 = 1.0 means perfect discrimination.
20 neurons per dataset: the 10 most target-predictive + 10 random alive. SAE: Top-K, M=256, K=8, Matryoshka
[32,256], on text-embedding-3-small.</p>
<h2>The three methods</h2>
<p><b>1. Baseline</b> — the repo's default pipeline: one gpt-5.2 interpreter call shown the top-10 activating + 10 random
zero-activation examples, fixed prompt. No iteration.</p>
<p><b>2. GEPA</b> — vanilla GEPA (official <code>gepa</code> package): evolves the description by reflecting on
misclassified texts from scored rollouts, Pareto selection over examples, best = argmax aggregate score. Seeded
with the baseline description. Sees scores and misclassified texts only — no corpus access.</p>
<p><b>3. Agent</b> — one OpenAI Agents-SDK agent (gpt-5.5) per neuron with (a) a persistent Python sandbox holding
all texts, the full activation matrix, and embeddings, where it writes its own analysis/sampling code, and (b) the
official scorer, callable on the official set or on any text sample the agent picks. Seeded with the baseline
description as its floor.</p>
<h3>The agent's environment</h3>
<table class='cands'>
<tr><th colspan='3' style='background:#eaf1fb'><span style='color:#4f8cd6'>Tool 1: run_python(code)</span> — persistent Python sandbox (state survives across calls), preloaded with:</th></tr>
<tr><th>name</th><th>type / shape</th><th>contents</th></tr>
<tr><td><code>texts</code></td><td>list[N]</td><td>all texts (N = 10,000 Yelp reviews / 20,000 Congress chunks)</td></tr>
<tr><td><code>act</code></td><td>float[N]</td><td>this neuron's activation per text</td></tr>
<tr><td><code>all_acts</code></td><td>float[N &times; 256]</td><td>all SAE neuron activations</td></tr>
<tr><td><code>emb</code></td><td>float[N &times; 1536]</td><td>unit-norm text embeddings</td></tr>
<tr><td><code>stars</code></td><td>float[N]</td><td>target variable (star rating / party)</td></tr>
<tr><td><code>top</code></td><td>int[N]</td><td>text indices sorted by activation, descending</td></tr>
<tr><td><code>official_pos, official_neg</code></td><td>list[int]</td><td>indices of the official eval set (50+50 / 100+100)</td></tr>
<tr><td><code>show(i, words=120)</code></td><td>function</td><td>print text i, truncated</td></tr>
<tr><td><code>np</code></td><td>module</td><td>numpy</td></tr>
<tr><th colspan='3' style='background:#fbeaf2'><span style='color:#d81b60'>Tool 2: score_descriptions(descriptions, indices=None)</span> — the official annotator (gpt-5-mini, repo prompt). indices=None scores the full official set (the reported metric); custom indices score any self-picked sample (labels = activation&gt;0). Budget 1,000 annotator calls per neuron; one call = one text &times; one description; cached pairs free; max 5 descriptions per call. Returns precision/recall/F1 + misclassified text indices.</th></tr>
</table>
<p>All three arms are scored on the identical official set per neuron. GEPA and the agent get the same
optimization budget (1,000 annotator calls per neuron; one call = one text &times; one description) and the same
output constraint: a single-sentence, objectively checkable property — no annotator directives, no multi-step
instructions. Annotator: gpt-5-mini throughout; it is the only model that ever produces the yes/no judgments
behind any score.</p>
<p><b>A fourth method</b> is the same agent with one change. Its official score comes from a held-out set of
documents that it can score but never read. Each dataset section below opens with a highlighted table that
compares all four methods on those held-out documents.</p>
"""


def main():
    body = HEAD + INTRO
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_done = []
    for name, art, n_test, blurb in DATASETS:
        if not os.path.isdir(art):
            continue
        data = collect(art, n_test)
        n_agent = sum("agent" in d for d in data.values())
        n_gepa = sum("gepa" in d for d in data.values())
        total_done.append(f"{name}: agent {n_agent}/{len(data)}, GEPA {n_gepa}/{len(data)}")
        body += f"<h2>{name}</h2>{blurb}"
        body += (f"<p class='statusbar'><b>Progress:</b> agent {n_agent}/{len(data)} neurons, "
                 f"GEPA {n_gepa}/{len(data)} neurons. Rows marked … are still running; "
                 f"this page regenerates automatically as neurons finish (last build {stamp}).</p>")
        body += f"<img src='data:image/png;base64,{bars_fig(data, name + ' — method comparison')}'/>"
        body += blind_holdout_section(art, n_test, data)
        svg = cost_accuracy_svg(data, n_test, f"{name}: cost vs F1 (hover a point for the neuron)", name.lower())
        if svg:
            body += svg
        tfig = trajectory_fig(data, name)
        if tfig:
            body += f"<img src='data:image/png;base64,{tfig}'/>"
        body += neuron_table(data, name.lower(), n_test)
        ver = load(os.path.join(art, "verification.json"))
        if ver:
            rows = ""
            for mkey, label in [("agent", "Agent"), ("gepa", "GEPA")]:
                if mkey not in ver:
                    continue
                gaps = [v["claimed_f1"] - v["fresh_f1"] for v in ver[mkey].values()]
                claimed = np.mean([v["claimed_f1"] for v in ver[mkey].values()])
                fresh = np.mean([v["fresh_f1"] for v in ver[mkey].values()])
                rows += (f"<tr><td>{label}</td><td>{claimed:.3f}</td><td>{fresh:.3f}</td>"
                         f"<td>{np.mean(gaps):+.4f}</td></tr>")
            body += ("<h3>Verification: fresh re-annotation of every final description</h3>"
                     "<p class='muted'>Each winning description was re-scored with brand-new gpt-5-mini "
                     "annotations (cache fully bypassed) on the same official set. A near-zero gap means the "
                     "claimed scores reflect real annotator behavior, not caching artifacts or selection luck; "
                     "the residual is annotator sampling noise.</p>"
                     "<table class='cands'><tr><th>method</th><th>claimed mean F1</th>"
                     f"<th>fresh mean F1</th><th>gap</th></tr>{rows}</table>")
        ev = evolution_blocks(data)
        if ev:
            body += "<h3>Candidate evolution (largest improvements)</h3>" + ev
    body += "</body></html>"
    out_path = os.path.join(BASE, "report.html")
    with open(out_path, "w") as f:
        f.write(body)
    print(f"wrote {out_path} ({'; '.join(total_done)})")


if __name__ == "__main__":
    main()
