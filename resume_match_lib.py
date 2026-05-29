"""Constraint logic for the resume->job matching demo (judge-free, regex-only).

Consolidates the parsing + 3-axis filter that the probes validated, so the demo
is self-contained and does not import the scratch/ probe scripts (which pull in
sentence_transformers / pandas at module import). Provenance:
  seniority/geo/location      <- scratch/probe_hard_constraints.py
  years/degree/cred/clearance <- scratch/gate_constraints.py
  ok_sen/ok_loc/ok_gate/rerank<- scratch/probe_3axis_rerank.py

The three hard-constraint axes (all high-confidence):
  sen   |resume_level - job_level| < 2          (ladder 0=intern..5=exec)
  loc   job remote OR geo-token overlap
  gate  resume passes every high-conf gate the job STATES (years>=req, degree>=req,
        holds all required creds). clearance/workauth are DETECTED but excluded
        from the filter (resume cannot disprove them).
"""

import json
import re

# --- seniority ladder: token -> level. Higher = more senior. ---
SENIORITY = [
    (re.compile(r"\b(intern|internship|co-?op)\b", re.I), 0),
    (re.compile(r"\b(junior|jr\.?|entry[- ]?level|graduate|trainee|apprentice)\b", re.I), 1),
    (re.compile(r"\b(senior|sr\.?|snr)\b", re.I), 3),
    (re.compile(r"\b(staff|principal|lead|architect)\b", re.I), 4),
    (
        re.compile(r"\b(director|head of|vp|vice president|chief|cto|ceo|cfo|coo|founder)\b", re.I),
        5,
    ),
]
MID = 2  # default when no modifier token present
SENIORITY_LABELS = {0: "intern", 1: "junior", 2: "mid", 3: "senior", 4: "staff/lead", 5: "exec"}


def seniority_of(text):
    """Max seniority level found in text; MID if none."""
    best = None
    for rx, lvl in SENIORITY:
        if rx.search(text or ""):
            best = lvl if best is None else max(best, lvl)
    return MID if best is None else best


# --- location parsing ---
STOP_GEO = {
    "remote",
    "united",
    "states",
    "usa",
    "us",
    "uk",
    "the",
    "of",
    "area",
    "greater",
    "metropolitan",
    "region",
    "and",
    "or",
    "multiple",
    "various",
}


def geo_tokens(s):
    """Lowercased alpha tokens from a location string, minus filler."""
    toks = re.findall(r"[a-zA-Z]+", (s or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in STOP_GEO}


def job_locations(d):
    locs = d.get("locations")
    if isinstance(locs, str):
        try:
            locs = json.loads(locs)
        except Exception:
            locs = [locs]
    return locs if isinstance(locs, list) else []


def job_is_remote(d):
    if str(d.get("remote")) == "True":
        return True
    return any("remote" in str(x).lower() for x in job_locations(d))


# --- years of experience required (job side) ---
_YEARS_RX = re.compile(
    r"\b(\d{1,2})\s*(?:\+|to|-|–)?\s*(\d{1,2})?\s*\+?\s*years?\b(?:\s*(?:of|of\s+relevant|of\s+professional)?)?\s*"
    r"(?:[\w/&,\- ]{0,40}?)\bexperience\b",
    re.I,
)


def job_years_req(text):
    """Largest 'N years ... experience' requirement in text, or None."""
    best = None
    for m in _YEARS_RX.finditer(text or ""):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        n = max(lo, hi)
        if 0 < n <= 25:  # ignore absurd matches ("2010 years")
            best = n if best is None else max(best, n)
    return best


_YEAR_TOK = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")  # 1980..2029, plausible employment years


def resume_years(experience):
    """Total professional span = latest - earliest employment year across all roles.

    Field order/date format vary, so collect every plausible 4-digit year plus map
    Present/Current -> 2025, and take max-min. Returns float years, or None.
    """
    if not experience:
        return None
    t = str(experience)
    years = [int(y) for y in _YEAR_TOK.findall(t)]
    if re.search(r"present|current|now\b", t, re.I):
        years.append(2025)  # corpus posted_at is 2025; treat ongoing role as "now"
    if not years:
        return None
    return float(max(years) - min(years))


# --- degree level: 0 none, 1 bachelor, 2 master, 3 phd ---
_DEG_JOB = [
    (
        re.compile(
            r"\b(ph\.?d|doctorate|doctoral)\b.{0,30}\b(requir|must)\b|"
            r"\b(requir|must)\b.{0,30}\b(ph\.?d|doctorate)\b",
            re.I,
        ),
        3,
    ),
    (
        re.compile(
            r"\bmaster'?s?\b.{0,30}\b(requir|must)\b|"
            r"\b(requir|must)\b.{0,30}\bmaster'?s?\b",
            re.I,
        ),
        2,
    ),
    (
        re.compile(
            r"\bbachelor'?s?\b.{0,30}\b(requir|must)\b|"
            r"\b(requir|must)\b.{0,30}\bbachelor'?s?\b|"
            r"\bdegree\s+(is\s+)?required\b",
            re.I,
        ),
        1,
    ),
]
_DEG_RESUME = [
    (re.compile(r"\b(ph\.?d|doctorate|doctoral|d\.?phil)\b", re.I), 3),
    (re.compile(r"\b(master'?s?|m\.?s\.?|m\.?b\.?a|m\.?eng|mca|m\.?a\.?)\b", re.I), 2),
    (re.compile(r"\b(bachelor'?s?|b\.?s\.?|b\.?a\.?|b\.?eng|b\.?tech|undergraduate)\b", re.I), 1),
]
DEGREE_LABELS = {0: "none", 1: "bachelor", 2: "master", 3: "phd"}


def job_degree_req(text):
    """Highest degree level explicitly *required* by the job, or 0."""
    t = text or ""
    for rx, lvl in _DEG_JOB:
        if rx.search(t):
            return lvl
    return 0


def resume_degree(education):
    """Highest degree level present in Education, or 0."""
    t = education or ""
    for rx, lvl in _DEG_RESUME:
        if rx.search(t):
            return lvl
    return 0


# --- named credentials (cert / professional license) ---
CRED_RX = {
    "cpa": re.compile(r"\bcpa\b", re.I),
    "cissp": re.compile(r"\bcissp\b", re.I),
    "pmp": re.compile(r"\bpmp\b", re.I),
    "cfa": re.compile(r"\bcfa\b", re.I),
    "ccna": re.compile(r"\bccn[ap]\b", re.I),
    "comptia": re.compile(r"\bcomptia|security\+|network\+\b", re.I),
    "six_sigma": re.compile(r"\bsix sigma\b", re.I),
    "aws_cert": re.compile(r"\baws certified\b", re.I),
    "azure_cert": re.compile(r"\bazure (certified|certification)\b", re.I),
    "gcp_cert": re.compile(r"\b(gcp|google cloud) (certified|certification)\b", re.I),
    "pe_license": re.compile(
        r"\b(pe license|professional engineer license|licensed professional engineer)\b", re.I
    ),
    "rn_license": re.compile(r"\b(rn license|registered nurse|nursing license)\b", re.I),
    "bar": re.compile(r"\b(bar admission|admitted to the bar|licensed attorney)\b", re.I),
    "cdl": re.compile(r"\bcdl\b|commercial driver", re.I),
}
_REQ_NEAR = re.compile(r"\b(requir|must have|must possess|mandatory|need(ed)?)\b", re.I)


def job_cred_gates(text):
    """Set of credential tokens the job appears to REQUIRE (not 'a plus')."""
    t = text or ""
    gates = set()
    for tok, rx in CRED_RX.items():
        m = rx.search(t)
        if not m:
            continue
        a, b = max(0, m.start() - 60), min(len(t), m.end() + 60)
        if _REQ_NEAR.search(t[a:b]):
            gates.add(tok)
    return gates


def resume_creds(skills, certs, experience):
    """Set of credential tokens the resume evidences."""
    blob = " ".join(str(x) for x in (skills, certs, experience) if x)
    return {tok for tok, rx in CRED_RX.items() if rx.search(blob)}


# === feature extraction (one record per resume / job) ===
def resume_features(row):
    """Build a feature dict from a resume parquet row (pandas Series or dict-like)."""

    def g(k):
        v = row.get(k) if hasattr(row, "get") else row[k]
        return "" if v is None else str(v)

    head = g("Headline").strip()
    exp = g("Experience").strip()
    exp_titles = " | ".join(exp.split("\n")[:2])[:200]
    sen = max(seniority_of(head), seniority_of(exp_titles))
    name = (g("FirstName") + " " + g("LastName")).strip()
    return {
        "name": name,
        "headline": head[:160],
        "loc": g("Location").strip(),
        "loc_tok": sorted(geo_tokens(g("Location"))),
        "text": g("text")[:2000],
        "seniority": sen,
        "years": resume_years(g("Experience")),
        "degree": resume_degree(g("Education")),
        "creds": sorted(resume_creds(g("Skills"), g("Certifications"), g("Experience"))),
    }


def features_from_text(text, loc=""):
    """Build a feature dict from free-form profile text (paste / .txt / LinkedIn PDF).

    Unlike resume_features (which reads structured parquet columns) there is only a
    text blob here, so:
      seniority -> highest level mentioned anywhere (a profile lists its top role)
      location  -> the optional `loc` field, else geo tokens scanned from the text
                   (permissive: unknown location tends to pass, matching the
                   gate-axis "unknown -> pass" convention rather than wrongly rejecting)
      years/degree/creds reuse the same regex extractors as the parquet path.
    """
    text = (text or "").strip()
    loc = (loc or "").strip()
    first_line = text.split("\n", 1)[0].strip() if text else ""
    loc_src = loc or text
    return {
        "name": "(your profile)",
        "headline": first_line[:160],
        "loc": loc,
        "loc_tok": sorted(geo_tokens(loc_src)),
        "text": text[:2000],
        "seniority": seniority_of(text),
        "years": resume_years(text),
        "degree": resume_degree(text),
        "creds": sorted(resume_creds(text, "", text)),
    }


def job_features(d):
    """Build a feature dict from a parsed metadata.jsonl record."""
    txt = d.get("text") or ""
    locs = job_locations(d)
    loc_tok = set().union(*[geo_tokens(str(x)) for x in locs]) if locs else set()
    return {
        "title": d.get("title", ""),
        "sen": seniority_of(d.get("title", "")),
        "remote": job_is_remote(d),
        "loc": "; ".join(str(x) for x in locs),
        "loc_tok": sorted(loc_tok),
        "years_req": job_years_req(txt),
        "degree_req": job_degree_req(txt),
        "cred_gates": sorted(job_cred_gates(txt)),
        "clearance": bool(CLEARANCE_RX.search(txt)),
        "workauth": bool(WORKAUTH_RX.search(txt)),
    }


# --- categorical, low-confidence-on-resume gates (detected, not filtered on) ---
CLEARANCE_RX = re.compile(
    r"\b(security clearance|ts/?sci|top secret|secret clearance|polygraph|"
    r"public trust|active clearance|dod clearance)\b",
    re.I,
)
WORKAUTH_RX = re.compile(
    r"\b(authoriz(?:ed|ation) to work|work authorization|must be (?:a )?(?:us|u\.s\.) citizen|"
    r"citizenship required|no sponsorship|without sponsorship|cannot provide sponsor|"
    r"not (?:able to )?(?:provide )?sponsor)\b",
    re.I,
)


# === 3-axis filter (operates on feature dicts, not raw records) ===
YEARS_SLACK = 0
CRED_LABELS = {
    "cpa": "CPA",
    "cissp": "CISSP",
    "pmp": "PMP",
    "cfa": "CFA",
    "ccna": "CCNA/CCNP",
    "comptia": "CompTIA",
    "six_sigma": "Six Sigma",
    "aws_cert": "AWS Cert",
    "azure_cert": "Azure Cert",
    "gcp_cert": "GCP Cert",
    "pe_license": "PE License",
    "rn_license": "RN License",
    "bar": "Bar Admission",
    "cdl": "CDL",
}


def axis_status(r, j):
    """Return per-axis pass/fail + human-readable reasons for a (resume, job) pair.

    r and j are feature dicts (loc_tok/creds/cred_gates are lists here).
    """
    r_loc = set(r["loc_tok"])
    j_loc = set(j["loc_tok"])
    r_creds = set(r["creds"])
    j_gates = set(j["cred_gates"])

    # seniority
    sen_gap = abs(r["seniority"] - j["sen"])
    sen_ok = sen_gap < 2
    if sen_ok:
        sen_reason = ""
    elif r["seniority"] > j["sen"]:
        sen_reason = (
            f"overqualified ({SENIORITY_LABELS[r['seniority']]} vs {SENIORITY_LABELS[j['sen']]})"
        )
    else:
        sen_reason = (
            f"underqualified ({SENIORITY_LABELS[r['seniority']]} vs {SENIORITY_LABELS[j['sen']]})"
        )

    # location
    loc_ok = j["remote"] or bool(r_loc & j_loc)
    loc_reason = "" if loc_ok else "location mismatch (not remote, no geo overlap)"

    # gate
    gate_ok = True
    gate_reasons = []
    if (
        j["years_req"] is not None
        and r["years"] is not None
        and r["years"] < j["years_req"] - YEARS_SLACK
    ):
        gate_ok = False
        gate_reasons.append(f"needs {j['years_req']} yrs, resume {int(r['years'])}")
    if j["degree_req"] > 0 and r["degree"] < j["degree_req"]:
        gate_ok = False
        gate_reasons.append(
            f"needs {DEGREE_LABELS[j['degree_req']]}, resume {DEGREE_LABELS[r['degree']]}"
        )
    missing_creds = j_gates - r_creds
    if missing_creds:
        gate_ok = False
        gate_reasons.append(
            "missing " + ", ".join(CRED_LABELS.get(c, c) for c in sorted(missing_creds))
        )

    return {
        "sen": {"ok": sen_ok, "reason": sen_reason},
        "loc": {"ok": loc_ok, "reason": loc_reason},
        "gate": {"ok": gate_ok, "reason": "; ".join(gate_reasons)},
        "all": sen_ok and loc_ok and gate_ok,
    }
