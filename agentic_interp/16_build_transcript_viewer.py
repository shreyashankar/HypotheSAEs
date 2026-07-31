"""Build ONE transcripts.html with a dataset/neuron picker.

Transcript data (agent JSON sidecars + GEPA candidate records) is embedded into
the page; rendering happens client-side. Thinking blocks render as markdown.
Deep-linkable via #dataset-nID.
"""
import os, sys, re, glob, json
import importlib

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))
from agentic_interp.harness import AGENT_RUN_DIRNAME, GEPA_RUN_DIRNAME
DATASETS = {"yelp": os.path.join(BASE, "artifacts"),
            "congress": os.path.join(BASE, "artifacts_congress")}

def build_instructions(ds, art, j):
    import agentic_interp.harness as H
    m10 = importlib.import_module("agentic_interp.10_agent_v5")
    recs = json.load(open(os.path.join(art, "train_texts.json")))
    baselines = json.load(open(os.path.join(art, "baseline_results.json")))
    bdesc = baselines.get(str(j), {}).get("description", "(none available)")
    return m10.INSTRUCTIONS_TEMPLATE.format(
        j=j, task_instructions=H._INSTRUCTIONS[ds],
        budget=m10.ANNOTATION_BUDGET, max_per_call=m10.MAX_DESCS_PER_CALL,
        n_total=len(recs), dataset_desc=H._DESCS[ds], baseline_desc=bdesc)


def collect_all():
    data = {}
    for ds, art in DATASETS.items():
        run_dir = os.path.join(art, AGENT_RUN_DIRNAME)
        if os.path.isdir(run_dir):
            key = f"{ds}_agent"
            data[key] = {}
            for p in glob.glob(os.path.join(run_dir, "transcript_n*.json")):
                j = re.search(r"transcript_n(\d+)\.json", p).group(1)
                rec = json.load(open(p))
                if not rec["transcript"] or rec["transcript"][0][0] != "SYSTEM PROMPT":
                    try:
                        sp = build_instructions(ds, art, j)
                        rec["transcript"] = ([["SYSTEM PROMPT", sp],
                                              ["USER", "Begin analyzing the neuron."]]
                                             + rec["transcript"])
                    except Exception as e:
                        print(f"  (no system prompt for {ds} n{j}: {e})")
                data[key][j] = rec
        hold_dir = os.path.join(art, "agent_runs_holdout")
        if os.path.isdir(hold_dir):
            key = f"{ds}_agent_holdout"
            data[key] = {}
            # Stage 20 always embeds the system prompt, so no rebuild fallback.
            for p in glob.glob(os.path.join(hold_dir, "transcript_n*.json")):
                j = re.search(r"transcript_n(\d+)\.json", p).group(1)
                data[key][j] = json.load(open(p))
        gepa_dir = os.path.join(art, GEPA_RUN_DIRNAME)
        if os.path.isdir(gepa_dir):
            key = f"{ds}_gepa"
            data[key] = {}
            for p in glob.glob(os.path.join(gepa_dir, "n*.json")):
                rec = json.load(open(p))
                j = str(rec["neuron"])
                best_i = rec.get("best_idx", 0)
                cands = rec.get("candidates", [])
                payload = {"candidates": cands, "val_scores": rec.get("val_scores", []),
                           "best_idx": best_i}
                data[key][j] = {
                    "claimed": cands[best_i] if cands else "",
                    "rationale": f"GEPA evolution: {len(cands)} candidates; best = #{best_i} by aggregate validation score.",
                    "transcript": [["GEPA CANDIDATES", json.dumps(payload)]]}
    return data


PAGE = """<!DOCTYPE html><html><head><meta charset='utf-8'><link rel='preconnect' href='https://fonts.googleapis.com'><link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap' rel='stylesheet'><title>Agent transcripts</title>
<style>
body { font-family: 'Inter', -apple-system, Helvetica, Arial, sans-serif; max-width: 980px; margin: 0 auto; padding: 1em; color: #222; line-height: 1.45; background: #fafafa; }
.topbar { position: sticky; top: 0; background: #fafafa; padding: 0.7em 0; border-bottom: 1px solid #e5e5e5; z-index: 5; }
select { font-size: 1em; padding: 0.25em 0.5em; border-radius: 6px; margin-right: 0.6em; }
.block { margin: 0.8em 0; border-radius: 10px; padding: 0.75em 1em; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.reasoning { background: #f7f4fb; color: #4a4458; }
.assistant { background: #f1f8f1; }
.toolcall  { background: #f2f6fc; }
.tooloutput{ background: #fdf9ee; }
pre { white-space: pre-wrap; margin: 0.4em 0 0; font-size: 0.82em; overflow-x: auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.lbl { font-weight: 700; font-size: 0.7em; letter-spacing: 0.08em; color: #999; text-transform: uppercase; display: block; margin-bottom: 0.25em; }
.hdr { background: #eef3fb; }
table.scores { border-collapse: collapse; width: 100%; margin-top: 0.4em; font-size: 0.85em; background: #fff; }
table.scores th, table.scores td { border: 1px solid #e6e6e6; padding: 4px 8px; text-align: left; }
table.scores th { background: #f4f6f8; }
.f1hi { background: #e7f5e7; font-weight: 600; }
.meta { color: #777; font-size: 0.82em; margin-top: 0.4em; }
.md p { margin: 0.3em 0; } .md ul { margin: 0.3em 0 0.3em 1.3em; padding: 0; } .md code { background: #f0f0f0; padding: 0 3px; border-radius: 3px; font-size: 0.9em; }
.md h1,.md h2,.md h3 { font-size: 1em; margin: 0.5em 0 0.2em; }
.viz { background: #fff; border-radius: 10px; padding: 0.6em 1em; margin: 0.8em 0; box-shadow: 0 1px 4px rgba(0,0,0,0.10); position: sticky; top: 52px; z-index: 4; }
.viz svg { display: block; width: 100%; }
.vlegend { font-size: 0.75em; color: #555; margin-top: 0.4em; }
.vlegend span.sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin: 0 4px 0 12px; vertical-align: -1px; }
</style></head><body>
<div class='topbar'>
  <b>Agent transcripts</b> &nbsp;
  <select id='ds'></select>
  <select id='nr'></select>
  <span class='meta' id='pathinfo'></span>
</div>
<div id='content'></div>
<script>
const DATA = __DATA__;
const ART = {
  yelp_agent: 'artifacts/agent_runs_strict', yelp_gepa: 'artifacts/gepa_runs_v5',
  yelp_agent_holdout: 'artifacts/agent_runs_holdout',
  congress_agent: 'artifacts_congress/agent_runs_strict', congress_gepa: 'artifacts_congress/gepa_runs_v5',
  congress_agent_holdout: 'artifacts_congress/agent_runs_holdout'
};

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function md(s) {
  let h = esc(s);
  h = h.replace(/^### (.*)$/gm, '<h3>$1</h3>').replace(/^## (.*)$/gm, '<h2>$1</h2>').replace(/^# (.*)$/gm, '<h1>$1</h1>');
  h = h.replace(/\\*\\*([^*]+)\\*\\*/g, '<b>$1</b>').replace(/(^|\\W)\\*([^*\\n]+)\\*(?=\\W|$)/g, '$1<i>$2</i>');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/^[-•] (.*)$/gm, '<li>$1</li>').replace(/(<li>.*<\\/li>\\n?)+/gs, m => '<ul>' + m + '</ul>');
  h = h.split(/\\n{2,}/).map(p => p.startsWith('<') ? p : '<p>' + p.replace(/\\n/g, '<br>') + '</p>').join('');
  return h;
}
function scoreTable(obj) {
  const res = obj.results || {};
  let rows = '';
  for (const [desc, m] of Object.entries(res)) {
    if (typeof m !== 'object' || m.f1 === undefined) continue;
    const cls = m.f1 >= 1.0 ? " class='f1hi'" : '';
    const nfp = (m.false_positive_idx || []).length, nfn = (m.false_negative_idx || []).length;
    rows += `<tr><td>${esc(desc)}</td><td${cls}>${m.f1.toFixed(3)}</td><td>${(m.precision||0).toFixed(2)}</td><td>${(m.recall||0).toFixed(2)}</td><td>${nfp}</td><td>${nfn}</td></tr>`;
  }
  if (!rows) return null;
  let meta = `<div class='meta'>${esc(String(obj.evaluated_on||'?'))} · budget: ${esc(String(obj.annotator_calls_spent||'?'))}`;
  if (obj.best_official_so_far) meta += ` · best so far: F1 ${obj.best_official_so_far.f1} — “${esc(String(obj.best_official_so_far.description||'').slice(0,90))}”`;
  return `<table class='scores'><tr><th>description</th><th>F1</th><th>prec</th><th>rec</th><th>#FP</th><th>#FN</th></tr>${rows}</table>${meta}</div>`;
}
const CAT = {
  system:   {label: 'system prompt',        color: '#9aa0a6', out: false},
  user:     {label: 'user',                 color: '#c5cad1', out: false},
  thinking: {label: 'thinking',             color: '#b39ddb', out: false},
  assistant:{label: 'assistant text',       color: '#66bb6a', out: false},
  code:     {label: 'run_python (code authored)', color: '#4f8cd6', out: false},
  pyout:    {label: 'run_python output',    color: '#4f8cd6', out: true},
  scall:    {label: 'score_descriptions (candidates)', color: '#d81b60', out: false},
  sout:     {label: 'score results',        color: '#d81b60', out: true}
};
function catSeq(rec) {
  const seq = []; let lastTool = '';
  rec.transcript.forEach(function(item, i) {
    const role = item[0], txt = item[1] || '';
    let c = null;
    if (role === 'SYSTEM PROMPT') c = 'system';
    else if (role === 'USER') c = 'user';
    else if (role === 'REASONING SUMMARY') c = 'thinking';
    else if (role === 'ASSISTANT') c = 'assistant';
    else if (role.startsWith('TOOL CALL')) {
      lastTool = role.includes('score_descriptions') ? 'score' : 'py';
      c = lastTool === 'score' ? 'scall' : 'code';
    } else if (role === 'TOOL OUTPUT') c = lastTool === 'score' ? 'sout' : 'pyout';
    if (c) seq.push({cat: c, chars: txt.length, idx: i, snip: txt.slice(0, 90)});
  });
  return seq;
}
function hatchDefs() {
  let d = '<defs>';
  for (const k in CAT) if (CAT[k].out)
    d += `<pattern id='h-${k}' patternUnits='userSpaceOnUse' width='6' height='6'>` +
         `<rect width='6' height='6' fill='${CAT[k].color}' fill-opacity='0.35'/>` +
         `<path d='M0 6 L6 0' stroke='${CAT[k].color}' stroke-width='1.6'/></pattern>`;
  return d + '</defs>';
}
function fillFor(c) { return CAT[c].out ? `url(#h-${c})` : CAT[c].color; }
function fmtK(n) { return n >= 1000 ? (n/1000).toFixed(1) + 'k' : String(n); }
const STATE_VARS = ['texts', 'act', 'all_acts', 'emb', 'stars', 'top', 'official_pos', 'official_neg', 'show', 'np'];
function varUsage(rec) {
  const counts = {};
  STATE_VARS.forEach(v => counts[v] = 0);
  let lastTool = '';
  rec.transcript.forEach(function(item) {
    const role = item[0];
    if (!role.startsWith('TOOL CALL') || !role.includes('run_python')) return;
    let code = item[1] || '';
    try { code = JSON.parse(code).code || code; } catch(e) {}
    STATE_VARS.forEach(function(v) {
      const m = code.match(new RegExp('\\\\b' + v + '\\\\b', 'g'));
      if (m) counts[v] += m.length;
    });
  });
  return counts;
}
function usageChips(rec) {
  const counts = varUsage(rec);
  const entries = Object.entries(counts).filter(e => e[1] > 0).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return '';
  const max = entries[0][1];
  const chips = entries.map(function(e) {
    const w = 8 + 52 * e[1] / max;
    return `<span style='display:inline-flex;align-items:center;margin:2px 8px 2px 0'>` +
           `<code style='font-size:0.78em'>${e[0]}</code>` +
           `<span style='display:inline-block;height:9px;width:${w.toFixed(0)}px;background:#4f8cd6;border-radius:2px;margin:0 4px'></span>` +
           `<span style='font-size:0.75em;color:#666'>&times;${e[1]}</span></span>`;
  }).join('');
  return `<div style='margin-top:0.5em'><span class='lbl'>sandbox state accessed (references in authored run_python code)</span>${chips}</div>`;
}
function buildViz(seq) {
  const W = 900, total = seq.reduce((a, s) => a + s.chars, 0) || 1;
  let x = 0, segs = '';
  seq.forEach(function(s) {
    const w = Math.max(1.5, s.chars / total * W);
    segs += `<rect x='${x.toFixed(1)}' y='4' width='${w.toFixed(1)}' height='26' fill='${fillFor(s.cat)}' ` +
            `class='seg' data-blk='blk${s.idx}' data-tip='#${s.idx} ${CAT[s.cat].label} — ${fmtK(s.chars)} chars (~${fmtK(Math.round(s.chars/4))} tok): ${s.snip.replace(/'/g, '&#39;')}' ` +
            `stroke='#fff' stroke-width='0.5' style='cursor:pointer'/>`;
    x += w;
  });
  const agg = {};
  seq.forEach(s => { agg[s.cat] = (agg[s.cat] || 0) + s.chars; });
  let ax = 0, aggSegs = '';
  for (const k in CAT) {
    if (!agg[k]) continue;
    const w = agg[k] / total * W;
    aggSegs += `<rect x='${ax.toFixed(1)}' y='4' width='${w.toFixed(1)}' height='26' fill='${fillFor(k)}' class='seg' ` +
               `data-tip='${CAT[k].label}: ${fmtK(agg[k])} chars (~${fmtK(Math.round(agg[k]/4))} tok, ${(agg[k]/total*100).toFixed(0)}%)' stroke='#fff' stroke-width='0.5'/>`;
    ax += w;
  }
  function sw(k) {
    return `<span class='sw' style='background:${CAT[k].out ? 'repeating-linear-gradient(45deg,' + CAT[k].color + '55 0 3px,' + CAT[k].color + ' 3px 4px)' : CAT[k].color}'></span>${CAT[k].label}`;
  }
  const GROUPS = [
    ['Fixed input', ['system', 'user']],
    ['Agent-authored (solid)', ['thinking', 'assistant', 'code', 'scall']],
    ['Tool-generated (hatched)', ['pyout', 'sout']]
  ];
  let legend = '';
  GROUPS.forEach(function(g) {
    const items = g[1].filter(k => agg[k]).map(sw).join('');
    if (items) legend += `<div><b style='color:#444'>${g[0]}:</b>${items}</div>`;
  });
  const CTX = 922000 * 4; // gpt-5.5 input window (~922k tokens) in chars
  const frac = Math.min(1, total / CTX);
  let winSegs = '', wx = 0;
  for (const k in CAT) {
    if (!agg[k]) continue;
    const w = agg[k] / total * (W * frac);
    winSegs += `<rect x='${wx.toFixed(1)}' y='4' width='${Math.max(0.5, w).toFixed(1)}' height='26' fill='${fillFor(k)}' class='seg' ` +
               `data-tip='${CAT[k].label}: ${fmtK(Math.round(agg[k]/4))} tok'/>`;
    wx += w;
  }
  winSegs += `<rect x='${wx.toFixed(1)}' y='4' width='${(W - wx).toFixed(1)}' height='26' fill='#f0f0f0' class='seg' ` +
             `data-tip='unused context window: ~${fmtK(Math.round((CTX - total)/4))} of 922k tokens free'/>` +
             `<text x='${Math.min(wx + 8, W - 220).toFixed(1)}' y='22' font-size='11' fill='#888'>${(frac*100).toFixed(1)}% of 922k-token window used</text>`;
  return `<div class='viz'><span class='lbl'>context timeline — each segment one operation, in order; width = share of context; click to jump</span>` +
         `<svg viewBox='0 0 ${W} 34'>${hatchDefs()}${segs}</svg>` +
         `<span class='lbl' style='margin-top:0.6em'>aggregate by type — total ~${fmtK(Math.round(total/4))} tokens · ` +
         `<a href='#' onclick='document.getElementById("aggshare").style.display = document.getElementById("aggshare").style.display === "none" ? "block" : "none"; document.getElementById("aggwin").style.display = document.getElementById("aggwin").style.display === "none" ? "block" : "none"; return false;' style='font-weight:400'>toggle: share of transcript ⇄ share of context window</a></span>` +
         `<div id='aggshare'><svg viewBox='0 0 ${W} 34'>${hatchDefs()}${aggSegs}</svg></div>` +
         `<div id='aggwin' style='display:none'><svg viewBox='0 0 ${W} 34'>${hatchDefs()}${winSegs}</svg></div>` +
         `<div class='vlegend'>${legend}</div></div>`;
}
const ENV = {
  yelp:     {n: '10,000', unit: 'Yelp reviews', ntest: '50 + 50'},
  congress: {n: '20,000', unit: 'Congressional speech chunks', ntest: '100 + 100'}
};
function envCard(ds, seq) {
  const e = ENV[ds.replace('_agent_holdout', '').replace('_agent', '').replace('_gepa', '')] || {n: '?', unit: 'texts', ntest: '?'};
  const isHold = ds.includes('agent_holdout');
  const nPy = seq.filter(s => s.cat === 'code').length;
  const nSc = seq.filter(s => s.cat === 'scall').length;
  const pre = [
    ['texts', `list[${e.n}]`, `all ${e.unit}`],
    ['act', `float[${e.n}]`, 'THIS neuron&#39;s activation per text'],
    ['all_acts', `float[${e.n} × 256]`, 'all SAE neuron activations'],
    ['emb', `float[${e.n} × 1536]`, 'unit-norm text embeddings'],
    ['stars', `float[${e.n}]`, 'target variable (rating / party)'],
    ['top', `int[${e.n}]`, 'text indices sorted by activation, descending'],
    ['show(i, words=120)', 'function', 'print text i, truncated'],
    ['np', 'module', 'numpy']
  ];
  if (!isHold) pre.splice(6, 0, ['official_pos / official_neg', 'list[int]', `official eval set indices (${e.ntest})`]);
  let rows = pre.map(r => `<tr><td><code>${r[0]}</code></td><td class='meta'>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
  const scorer = isHold
    ? `<code>indices=None</code> scores the held-out evaluation set (${e.ntest}) and returns only overall precision, recall, and F1; the held-out documents are not readable. Custom indices score any self-picked sample from the readable corpus (labels = activation&gt;0), with misclassified indices returned. `
    : `<code>indices=None</code> scores the full official set (the reported metric); custom indices score any self-picked sample (labels = activation&gt;0). `;
  return `<details class='block toolcall' style='margin-top:0.8em'><summary style='cursor:pointer'>` +
    `<span class='lbl' style='display:inline'>agent environment — 2 tools + preloaded sandbox</span>` +
    `<span class='meta'> · used ${nPy}× run_python, ${nSc}× score_descriptions</span></summary>` +
    `<div style='margin-top:0.5em'><b style='color:#4f8cd6'>run_python(code)</b> — persistent Python sandbox; state survives across calls. Preloaded variables:` +
    `<table class='scores' style='margin:0.4em 0'><tr><th>name</th><th>type / shape</th><th>contents</th></tr>${rows}</table>` +
    `<b style='color:#d81b60'>score_descriptions(descriptions, indices=None)</b> — the official annotator (gpt-5-mini, repo prompt). ` +
    scorer +
    `Budget: 1,000 annotator calls; one call = one text × one description; cached pairs free; max 5 descriptions per call.</div></details>`;
}
function render(ds, nr) {
  const rec = (DATA[ds]||{})[nr];
  const c = document.getElementById('content');
  document.getElementById('pathinfo').textContent = `${ART[ds] || '?'}/${ds.includes('gepa') ? 'n' : 'transcript_n'}${nr}.json`;
  if (!rec) { c.innerHTML = '<p>No transcript.</p>'; return; }
  const seq = catSeq(rec);
  let out = buildViz(seq);
  if (!ds.includes('gepa')) out = out.slice(0, -6) + usageChips(rec) + '</div>';
  out += `<div class='block hdr'><span class='lbl'>final description</span><b>${esc(rec.claimed||'(none)')}</b><br><span class='lbl' style='margin-top:0.5em'>rationale</span>${esc(rec.rationale||'')}</div>`;
  let _bi = -1;
  for (const [role, txt] of rec.transcript) {
    _bi += 1;
    const BID = ` id='blk${_bi}'`;
    if (role === 'SYSTEM PROMPT') {
      out += `<div${BID} class='block toolcall'><span class='lbl'>system prompt (rendered for this neuron)</span><details><summary>show full prompt</summary><pre>${esc(txt)}</pre></details></div>`;
    } else if (role === 'USER') {
      out += `<div${BID} class='block hdr'><span class='lbl'>user</span>${esc(txt)}</div>`;
    } else if (role === 'REASONING SUMMARY') {
      out += `<div${BID} class='block reasoning'><span class='lbl'>thinking</span><div class='md'>${md(txt)}</div></div>`;
    } else if (role === 'ASSISTANT') {
      let body;
      try { const o = JSON.parse(txt);
            body = o.best_description ? `<span class='lbl'>returned description</span><b>${esc(o.best_description)}</b><br><span class='lbl' style='margin-top:0.5em'>rationale</span>${esc(o.rationale||'')}` : null; } catch(e) { body = null; }
      out += `<div${BID} class='block assistant'><span class='lbl'>assistant</span>${body || "<div class='md'>" + md(txt) + '</div>'}</div>`;
    } else if (role.startsWith('TOOL CALL')) {
      const name = role.replace('TOOL CALL: ', '');
      let inner = `<pre>${esc(txt)}</pre>`;
      try { const a = JSON.parse(txt);
        if (name === 'run_python') inner = `<pre>${esc(a.code||txt)}</pre>`;
        else if (name === 'score_descriptions') {
          const tag = a.indices ? `custom sample (${a.indices.length} texts)`
                                : (ds.includes('agent_holdout') ? 'held-out evaluation set (not readable)' : 'official set');
          inner = `<div class='meta'>${tag}</div><ul>` + (a.descriptions||[]).map(d => `<li>${esc(d)}</li>`).join('') + '</ul>';
        }
      } catch(e) { if (name === 'run_python') inner = `<pre>${esc(txt)}</pre>`;
                   else if (name === 'score_descriptions' && txt.startsWith('[')) inner = `<pre>${esc(txt)}</pre>`; }
      out += `<div${BID} class='block toolcall'><span class='lbl'>tool call — ${esc(name)}</span>${inner}</div>`;
    } else if (role === 'GEPA CANDIDATES') {
      try {
        const g = JSON.parse(txt);
        let rows = '';
        g.candidates.forEach((d, i) => {
          const v = g.val_scores[i];
          const mark = i === g.best_idx ? ' ★ best' : (i === 0 ? ' (seed = baseline)' : '');
          const cls = i === g.best_idx ? " class='f1hi'" : '';
          rows += `<tr><td>${i}${mark}</td><td${cls}>${v !== undefined ? v.toFixed(3) : '?'}</td><td>${esc(d)}</td></tr>`;
        });
        out += `<div class='block tooloutput'><span class='lbl'>candidate evolution (score = mean per-example accuracy on official set)</span><table class='scores'><tr><th>#</th><th>val score</th><th>description</th></tr>${rows}</table></div>`;
      } catch(e) { out += `<div class='block tooloutput'><pre>${esc(txt)}</pre></div>`; }
    } else if (role === 'TOOL OUTPUT') {
      let inner = null;
      try { const o = JSON.parse(txt); inner = scoreTable(o) || `<pre>${esc(JSON.stringify(o, null, 2).slice(0,5000))}</pre>`; } catch(e) {}
      out += `<div${BID} class='block tooloutput'><span class='lbl'>tool output</span>${inner || `<pre>${esc(txt)}</pre>`}</div>`;
    }
  }
  c.innerHTML = out;
}
function armViz() {
  let tip = document.getElementById('viztip');
  if (!tip) {
    tip = document.createElement('div'); tip.id = 'viztip';
    tip.style.cssText = 'position:fixed;display:none;background:#222;color:#fff;padding:4px 9px;border-radius:5px;font-size:12px;pointer-events:none;z-index:99;max-width:480px';
    document.body.appendChild(tip);
  }
  document.querySelectorAll('rect.seg').forEach(function(r) {
    r.addEventListener('mousemove', function(e) {
      tip.textContent = r.getAttribute('data-tip');
      tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 500) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
      tip.style.display = 'block';
    });
    r.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
    r.addEventListener('click', function() {
      const b = r.getAttribute('data-blk');
      if (b) { const el = document.getElementById(b); if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'}); }
    });
  });
}
function init() {
  const dsSel = document.getElementById('ds'), nrSel = document.getElementById('nr');
  dsSel.innerHTML = Object.keys(DATA).map(d => `<option>${d}</option>`).join('');
  function fillN() {
    const ds = dsSel.value;
    nrSel.innerHTML = Object.keys(DATA[ds]||{}).sort((a,b)=>a-b).map(n => `<option value='${n}'>neuron ${n}</option>`).join('');
  }
  function go() { render(dsSel.value, nrSel.value); armViz(); location.hash = `${dsSel.value}-n${nrSel.value}`; }
  dsSel.onchange = () => { fillN(); go(); };
  nrSel.onchange = go;
  const m = location.hash.match(/#(\\w+)-n(\\d+)/);
  if (m && DATA[m[1]] && DATA[m[1]][m[2]]) { dsSel.value = m[1]; fillN(); nrSel.value = m[2]; }
  else fillN();
  render(dsSel.value, nrSel.value);
  armViz();
}
init();
</script></body></html>"""


def main():
    data = collect_all()
    n = sum(len(v) for v in data.values())
    payload = json.dumps(data).replace("</script", "<\\/script")
    html_out = PAGE.replace("__DATA__", payload)
    out = os.path.join(BASE, "transcripts.html")
    with open(out, "w") as f:
        f.write(html_out)
    print(f"wrote {out} ({n} transcripts, {len(html_out)//1024} KB)")


if __name__ == "__main__":
    main()
