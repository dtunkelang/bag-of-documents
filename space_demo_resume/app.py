#!/usr/bin/env python3
"""Resume -> job matching demo: browse/search 6.9k synthetic resumes, see the top
jobs for each, side-by-side as raw cosine vs a 3-axis constraint-aware re-rank.

The payoff this surfaces: dense cosine retrieval is constraint-blind. The probe
measured constraint-correct top-1 at 12.5% for raw cosine vs 95.5% after filtering
on seniority + location + qualification gates (years/degree/cred). This demo makes
that visible — each job carries sen/loc/gate badges, and the right column drops the
jobs that violate a hard constraint, promoting the highest-cosine survivor.

Caches (built by precompute_resume_match.py) make boot fast; no model is loaded at
serve time because resume vectors are precomputed.

Run:  .venv/bin/python demo_resume_match.py            # http://127.0.0.1:7863
"""

import argparse
import html
import json
import os
import pickle
import re
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import snapshot_download

import resume_match_lib as L

# Companion dataset holds the catalog vecs + metadata + precomputed resume caches.
# (Space file-size limits make this the cleanest split; mirrors the jobs-demo Space.)
DATASET_REPO = "dtunkelang/resume-job-match"
DATA_FILES = [
    "e5_base_catalog.vecs.fp16.npy",
    "metadata.jsonl",
    "resume_vecs.fp16.npy",
    "resume_records.json",
    "job_features.pkl",
    "job_offsets.npy",
]

# Paths are resolved at startup (see lifespan) once the dataset is downloaded.
CATALOG = META = RESUME_VECS = RESUME_RECS = JOB_FEATS = JOB_OFFSETS = None


def download_data() -> str:
    local = os.environ.get("LOCAL_DATA_DIR")
    if local:
        print(f"using LOCAL_DATA_DIR={local}", flush=True)
        return local
    print(f"snapshot_download from {DATASET_REPO}...", flush=True)
    return snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", allow_patterns=DATA_FILES)


POOL = 50  # candidate pool depth (matches the validated probe)
TOP_N = 10

R = {}  # loaded resources


@asynccontextmanager
async def lifespan(app):
    global CATALOG, META, RESUME_VECS, RESUME_RECS, JOB_FEATS, JOB_OFFSETS
    data_dir = download_data()
    CATALOG = os.path.join(data_dir, "e5_base_catalog.vecs.fp16.npy")
    META = os.path.join(data_dir, "metadata.jsonl")
    RESUME_VECS = os.path.join(data_dir, "resume_vecs.fp16.npy")
    RESUME_RECS = os.path.join(data_dir, "resume_records.json")
    JOB_FEATS = os.path.join(data_dir, "job_features.pkl")
    JOB_OFFSETS = os.path.join(data_dir, "job_offsets.npy")
    miss = [
        p
        for p in (CATALOG, META, RESUME_VECS, RESUME_RECS, JOB_FEATS, JOB_OFFSETS)
        if not os.path.exists(p)
    ]
    if miss:
        raise SystemExit("missing data file(s):\n  " + "\n  ".join(miss))
    t0 = time.time()
    print("loading catalog vecs (this loads ~1GB into RAM)...", flush=True)
    R["cat"] = np.load(CATALOG, mmap_mode="r").astype(np.float32)
    print(f"  catalog: {R['cat'].shape} {R['cat'].dtype}", flush=True)
    R["res_vecs"] = np.load(RESUME_VECS).astype(np.float32)
    with open(RESUME_RECS) as fh:
        R["res_recs"] = json.load(fh)
    with open(JOB_FEATS, "rb") as fh:
        R["job_feats"] = pickle.load(fh)
    R["job_offsets"] = np.load(JOB_OFFSETS)
    if R["res_vecs"].shape[0] != len(R["res_recs"]):
        raise SystemExit("resume vecs / records length mismatch")
    if R["cat"].shape[0] != len(R["job_feats"]):
        raise SystemExit("catalog / job_features length mismatch")
    # precompute a lowercased search blob per resume for the browse filter
    R["res_blob"] = [
        (r["name"] + " " + r["headline"] + " " + r["loc"] + " " + r["text"]).lower()
        for r in R["res_recs"]
    ]
    print(
        f"ready: {len(R['res_recs']):,} resumes, {len(R['job_feats']):,} jobs "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    yield
    R.clear()


app = FastAPI(title="Resume -> Job Matching Demo", lifespan=lifespan)

_WS_RUN = re.compile(r"[ \t]+")
_NL_RUN = re.compile(r"\n{3,}")


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    s = _WS_RUN.sub(" ", s)
    s = _NL_RUN.sub("\n\n", s)
    return s.strip()


def _job_brief(idx: int, cos: float, status: dict) -> dict:
    j = R["job_feats"][idx]
    return {
        "idx": idx,
        "title": j["title"],
        "loc": j["loc"],
        "remote": bool(j["remote"]),
        "seniority": L.SENIORITY_LABELS[j["sen"]],
        "years_req": j["years_req"],
        "degree_req": L.DEGREE_LABELS[j["degree_req"]] if j["degree_req"] else None,
        "cred_gates": [L.CRED_LABELS.get(c, c) for c in j["cred_gates"]],
        "clearance": bool(j["clearance"]),
        "workauth": bool(j["workauth"]),
        "cosine": round(float(cos), 4),
        "axes": status,
    }


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Resume &rarr; Job Matching</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1280px; margin: 24px auto; padding: 0 16px; color: #222; }
h1 { font-size: 1.35em; margin-bottom: 6px; }
.subtle { color: #777; font-size: 0.9em; margin-bottom: 16px; }
.layout { display: grid; grid-template-columns: 340px 1fr; gap: 18px; align-items: start; }
.col-resumes { border: 1px solid #ddd; border-radius: 6px; background: #fff; }
#resume-search { width: 100%; padding: 8px 12px; font-size: 1em; border: none; border-bottom: 1px solid #ddd; box-sizing: border-box; border-radius: 6px 6px 0 0; }
#resume-list { max-height: 78vh; overflow-y: auto; }
.rcard { padding: 9px 12px; border-bottom: 1px dotted #eee; cursor: pointer; }
.rcard:hover { background: #f6f9fe; }
.rcard.active { background: #eef4fb; border-left: 3px solid #2b6cb0; }
.rcard .nm { font-weight: 600; font-size: 0.95em; }
.rcard .hl { color: #444; font-size: 0.86em; margin-top: 2px; }
.rcard .lc { color: #888; font-size: 0.8em; margin-top: 2px; }
.rcard .sl { display:inline-block; font-size:0.72em; color:#2b6cb0; background:#e8f0fb; border-radius:8px; padding:1px 7px; margin-top:3px; }
.col-match { min-height: 200px; }
.empty { color: #999; padding: 40px; text-align: center; }
.rsum { border: 1px solid #ddd; border-radius: 6px; padding: 12px 14px; background: #fafbfc; margin-bottom: 14px; }
.rsum .nm { font-weight: 600; font-size: 1.05em; }
.rsum .hl { color: #444; margin: 3px 0; }
.rsum .facts { color: #555; font-size: 0.85em; margin-top: 6px; }
.rsum .facts b { color:#222; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.panel { border: 1px solid #ddd; border-radius: 6px; background: #fff; }
.panel h3 { margin: 0; padding: 9px 12px; font-size: 0.9em; border-bottom: 1px solid #eee; }
.panel.cos h3 { background: #f4eee8; color: #6b4a18; }
.panel.flt h3 { background: #e8f4ec; color: #186537; }
.panel .note { font-size: 0.78em; color: #999; padding: 6px 12px; border-bottom: 1px dotted #eee; }
.job { padding: 9px 12px; border-bottom: 1px dotted #eee; cursor: pointer; }
.job:hover { background: #fafafa; }
.job .jt { font-weight: 500; font-size: 0.92em; }
.job .jm { color: #777; font-size: 0.8em; margin-top: 2px; }
.job .jm .sep { color: #ccc; padding: 0 5px; }
.job .badges { margin-top: 5px; }
.b { display: inline-block; font-size: 0.72em; padding: 1px 7px; border-radius: 9px; margin-right: 5px; }
.b.ok { background: #e6f4ea; color: #1a7a3a; border: 1px solid #b6dec4; }
.b.bad { background: #fbe9e7; color: #b3261e; border: 1px solid #f0c2bd; }
.b.warn { background: #fff4e5; color: #8a5a00; border: 1px solid #f0d9a8; }
.cos-num { color: #555; font-variant-numeric: tabular-nums; font-size: 0.8em; float: right; }
.jobdetail { margin-top: 7px; padding: 9px 11px; background: #f7f7f9; border-left: 3px solid #c4c4cc; border-radius: 3px; white-space: pre-wrap; color: #333; font-size: 0.84em; line-height: 1.4; max-height: 320px; overflow-y: auto; }
.jobdetail.loading { color: #888; font-style: italic; }
</style></head>
<body>
<h1>Resume &rarr; Job Matching</h1>
<div class="subtle">
  __N_RES__ synthetic resumes &middot; __N_JOB__ jobs (e5-base-v2) &middot; click a resume to see its top jobs.
  Left column is <b>raw cosine</b>; right column applies a <b>3-axis hard-constraint filter</b>
  (seniority &middot; location &middot; qualification gates) and promotes the best-cosine survivor.
</div>
<div class="layout">
  <div class="col-resumes">
    <input id="resume-search" placeholder="filter resumes (name, headline, location, skills)..." autocomplete="off" />
    <div id="resume-list"><div class="empty">loading...</div></div>
  </div>
  <div class="col-match">
    <div id="match"><div class="empty">&larr; pick a resume to match</div></div>
  </div>
</div>
<script>
let RID = null;
let searchTimer = null;
function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

async function loadResumes(q){
  const r = await fetch('/api/resumes?q=' + encodeURIComponent(q||'') + '&limit=200');
  const d = await r.json();
  const box = document.getElementById('resume-list');
  if(!d.resumes.length){ box.innerHTML = '<div class="empty">no resumes match</div>'; return; }
  box.innerHTML = d.resumes.map(rr =>
    `<div class="rcard" data-rid="${rr.rid}" onclick="pick(${rr.rid})">
       <div class="nm">${esc(rr.name)}</div>
       <div class="hl">${esc(rr.headline)}</div>
       <div class="lc">${esc(rr.loc)}</div>
       <span class="sl">${esc(rr.seniority)}</span>
     </div>`).join('');
  if(d.truncated){ box.innerHTML += `<div class="note" style="padding:8px 12px;color:#999;font-size:0.78em">showing first ${d.resumes.length} of ${d.total} &mdash; refine the filter</div>`; }
  highlight();
}
function highlight(){
  document.querySelectorAll('.rcard').forEach(el =>
    el.classList.toggle('active', parseInt(el.dataset.rid) === RID));
}

function badge(name, ax){
  const cls = ax.ok ? 'ok' : 'bad';
  const mark = ax.ok ? '\\u2713' : '\\u2717';
  const tip = ax.reason ? ' \\u2014 ' + ax.reason : '';
  return `<span class="b ${cls}" title="${esc(ax.reason)}">${name} ${mark}${ax.ok?'':esc(tip)}</span>`;
}
function jobRow(j){
  const m = [];
  m.push(j.remote ? '\\ud83c\\udf10 remote' : esc(j.loc||'(no location)'));
  m.push('level: ' + esc(j.seniority));
  if(j.years_req!=null) m.push('needs ' + j.years_req + ' yrs');
  if(j.degree_req) m.push('needs ' + esc(j.degree_req));
  if(j.cred_gates && j.cred_gates.length) m.push('needs ' + j.cred_gates.map(esc).join(', '));
  const extra = [];
  if(j.clearance) extra.push('<span class="b warn" title="security clearance stated (not resume-checkable)">clearance</span>');
  if(j.workauth) extra.push('<span class="b warn" title="work-authorization stated (not resume-checkable)">work-auth</span>');
  return `<div class="job" onclick="toggleJob(${j.idx}, this)">
    <span class="cos-num">cos ${j.cosine.toFixed(3)}</span>
    <div class="jt">${esc(j.title)}</div>
    <div class="jm">${m.join('<span class="sep">&middot;</span>')}</div>
    <div class="badges">${badge('sen', j.axes.sen)}${badge('loc', j.axes.loc)}${badge('gate', j.axes.gate)}${extra.join('')}</div>
  </div>`;
}
async function toggleJob(idx, el){
  let ex = el.querySelector('.jobdetail');
  if(ex){ ex.remove(); return; }
  const div = document.createElement('div');
  div.className = 'jobdetail loading'; div.textContent = 'loading...';
  el.appendChild(div);
  try{
    const r = await fetch('/api/job_detail?idx=' + idx);
    const d = await r.json();
    div.classList.remove('loading');
    div.textContent = d.description || '(no description)';
  }catch(e){ div.classList.remove('loading'); div.textContent = '(failed to load)'; }
}

async function pick(rid){
  RID = rid; highlight();
  const box = document.getElementById('match');
  box.innerHTML = '<div class="empty">matching...</div>';
  const r = await fetch('/api/match?rid=' + rid);
  const d = await r.json();
  const rs = d.resume;
  const facts = [];
  facts.push('level: <b>' + esc(rs.seniority) + '</b>');
  if(rs.years!=null) facts.push('experience: <b>' + rs.years + ' yrs</b>');
  facts.push('degree: <b>' + esc(rs.degree) + '</b>');
  if(rs.creds && rs.creds.length) facts.push('creds: <b>' + rs.creds.map(esc).join(', ') + '</b>');
  const note = d.filtered_count < d.pool_n
    ? `${d.filtered_count} of top-${d.pool_n} pass all 3 axes`
    : `all top-${d.pool_n} pass`;
  box.innerHTML = `
    <div class="rsum">
      <div class="nm">${esc(rs.name)}</div>
      <div class="hl">${esc(rs.headline)}</div>
      <div class="lc" style="color:#888;font-size:0.85em">${esc(rs.loc)}</div>
      <div class="facts">${facts.join('<span style="color:#ccc;padding:0 6px">&middot;</span>')}</div>
    </div>
    <div class="panels">
      <div class="panel cos">
        <h3>Raw cosine (constraint-blind)</h3>
        <div class="note">nearest jobs by embedding similarity &mdash; ignores hard constraints</div>
        ${d.cosine.map(jobRow).join('') || '<div class="empty">none</div>'}
      </div>
      <div class="panel flt">
        <h3>3-axis constraint filter</h3>
        <div class="note">${note} &middot; best-cosine survivor first${d.filtered_count===0?' (none qualified &mdash; cosine top-1 fallback)':''}</div>
        ${(d.filtered.length?d.filtered:d.cosine.slice(0,1)).map(jobRow).join('')}
      </div>
    </div>`;
}

document.getElementById('resume-search').addEventListener('input', e => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadResumes(e.target.value.trim()), 160);
});
loadResumes('');
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE.replace("__N_RES__", f"{len(R['res_recs']):,}").replace(
        "__N_JOB__", f"{len(R['job_feats']):,}"
    )


@app.get("/api/resumes")
def api_resumes(q: str = Query(""), limit: int = Query(200)):
    recs = R["res_recs"]
    blob = R["res_blob"]
    ql = q.strip().lower()
    if ql:
        terms = ql.split()
        idxs = [i for i, b in enumerate(blob) if all(t in b for t in terms)]
    else:
        idxs = range(len(recs))
    total = len(idxs) if ql else len(recs)
    out = []
    for i in idxs:
        r = recs[i]
        out.append(
            {
                "rid": r["rid"],
                "name": r["name"] or "(unnamed)",
                "headline": r["headline"],
                "loc": r["loc"],
                "seniority": L.SENIORITY_LABELS[r["seniority"]],
            }
        )
        if len(out) >= limit:
            break
    return JSONResponse({"resumes": out, "total": total, "truncated": total > len(out)})


# rid -> row position in res_recs (rid is the original parquet index, not contiguous)
def _row_for_rid(rid: int):
    for pos, r in enumerate(R["res_recs"]):
        if r["rid"] == rid:
            return pos, r
    return None, None


@app.get("/api/match")
def api_match(rid: int = Query(...), pool: int = Query(POOL), n: int = Query(TOP_N)):
    pos, r = _row_for_rid(rid)
    if r is None:
        return JSONResponse({"error": "rid not found"}, status_code=404)
    qv = R["res_vecs"][pos]
    sims = R["cat"] @ qv  # (n_jobs,)
    pool = min(pool, sims.shape[0])
    cand = np.argpartition(-sims, pool - 1)[:pool]
    order = cand[np.argsort(-sims[cand])]

    cosine_list = []
    filtered_list = []
    filtered_count = 0
    for j in order:
        j = int(j)
        st = L.axis_status(r, R["job_feats"][j])
        brief = _job_brief(j, sims[j], st)
        if len(cosine_list) < n:
            cosine_list.append(brief)
        if st["all"]:
            filtered_count += 1
            if len(filtered_list) < n:
                filtered_list.append(brief)

    return JSONResponse(
        {
            "resume": {
                "name": r["name"] or "(unnamed)",
                "headline": r["headline"],
                "loc": r["loc"],
                "seniority": L.SENIORITY_LABELS[r["seniority"]],
                "years": int(r["years"]) if r["years"] is not None else None,
                "degree": L.DEGREE_LABELS[r["degree"]],
                "creds": [L.CRED_LABELS.get(c, c) for c in r["creds"]],
            },
            "pool_n": pool,
            "filtered_count": filtered_count,
            "cosine": cosine_list,
            "filtered": filtered_list,
        }
    )


@app.get("/api/job_detail")
def api_job_detail(idx: int = Query(...)):
    offsets = R["job_offsets"]
    if idx < 0 or idx >= len(offsets):
        return JSONResponse({"error": "idx out of range"}, status_code=404)
    with open(META, "rb") as f:
        f.seek(int(offsets[idx]))
        line = f.readline()
    rec = json.loads(line)
    return JSONResponse(
        {
            "idx": idx,
            "title": _clean_text(rec.get("title") or ""),
            "description": _clean_text(rec.get("description") or ""),
            "posted_at": rec.get("posted_at") or "",
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
