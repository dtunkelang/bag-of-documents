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
            r"\b(ph\.?d|doctorate|doctoral)\b.{0,30}\b(requir\w*|must)\b|"
            r"\b(requir\w*|must)\b.{0,30}\b(ph\.?d|doctorate)\b",
            re.I,
        ),
        3,
    ),
    (
        re.compile(
            r"\bmaster'?s?\b.{0,30}\b(requir\w*|must)\b|"
            r"\b(requir\w*|must)\b.{0,30}\bmaster'?s?\b",
            re.I,
        ),
        2,
    ),
    (
        re.compile(
            r"\bbachelor'?s?\b.{0,30}\b(requir\w*|must)\b|"
            r"\b(requir\w*|must)\b.{0,30}\bbachelor'?s?\b|"
            r"\bdegree\s+(is\s+)?required\b",
            re.I,
        ),
        1,
    ),
]
_DEG_RESUME = [
    # period-required forms (m\.d / j\.d) avoid colliding with state codes (e.g. "Baltimore, MD")
    (
        re.compile(
            r"\b(ph\.?d|doctorate|doctoral|d\.?phil|pharm\.?d|dnp|ed\.?d|psy\.?d|dds|dvm|m\.d|j\.d)\b",
            re.I,
        ),
        3,
    ),
    (
        re.compile(
            r"\b(master'?s?|m\.?s\.?|m\.?b\.?a|m\.?eng|mca|m\.?a\.?|msn|mph|mpa|msw|mfa)\b",
            re.I,
        ),
        2,
    ),
    (
        re.compile(
            r"\b(bachelor'?s?|b\.?s\.?|b\.?a\.?|b\.?eng|b\.?tech|undergraduate|bsn|bba|bfa|bsw)\b",
            re.I,
        ),
        1,
    ),
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
_REQ_NEAR = re.compile(r"\b(requir\w*|must have|must possess|mandatory|need(ed)?)\b", re.I)


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
        "country": loc_country(g("Location")),
        "text": g("text")[:2000],
        "seniority": sen,
        "years": resume_years(g("Experience")),
        "degree": resume_degree(g("Education")),
        "creds": sorted(resume_creds(g("Skills"), g("Certifications"), g("Experience"))),
    }


# --- free-text / LinkedIn-PDF parsing helpers (real resumes are unstructured and,
#     for the LinkedIn "Save to PDF" export, sidebar-first; the parquet extractors
#     above assume clean structured fields, so the demo's upload path needs these). ---
_SIDEBAR_LABELS = {
    "contact",
    "top skills",
    "languages",
    "summary",
    "experience",
    "education",
    "certifications",
    "honors-awards",
    "publications",
    "skills",
}
_ENTRY_RX = re.compile(
    r"\b(aspiring|recent graduate|new grad(uate)?|entry[- ]?level|undergraduate|"
    r"(currently |actively )?seeking|looking for (a |an |my )?(first |new )?(role|job|position|opportunit))\b",
    re.I,
)
_MONTH_YEAR = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(19[89]\d|20[0-2]\d)\b",
    re.I,
)
_EDU_KW = re.compile(
    r"\b(universit|college|bachelor|master|ph\.?d|b\.?s\.?|b\.?a\.?|b\.?tech|m\.?s\.?|"
    r"m\.?b\.?a|high school|diploma|gpa)\b",
    re.I,
)


def _looks_like_name(s):
    toks = s.split()
    return (
        1 <= len(toks) <= 4
        and len(s) <= 40
        and not any(ch.isdigit() for ch in s)
        and all(t[:1].isupper() for t in toks)
    )


def _parse_identity(text):
    """Best-effort (name, headline, location) from free text. For a LinkedIn PDF the
    layout is: ...sidebar... / Name / Headline / Location / Summary / ...; so when we
    detect that layout we read the three lines that precede the 'Summary' heading."""
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    low = [ln.lower() for ln in lines]
    name = headline = parsed_loc = ""
    is_linkedin = "linkedin.com/in" in (text or "").lower() or "top skills" in low
    if is_linkedin and "summary" in low:
        i = low.index("summary")
        if i >= 1 and low[i - 1] not in _SIDEBAR_LABELS:
            parsed_loc = lines[i - 1]
        if i >= 2 and low[i - 2] not in _SIDEBAR_LABELS:
            headline = lines[i - 2]
        if i >= 3 and _looks_like_name(lines[i - 3]):
            name = lines[i - 3]
    if not headline:
        for ln, l in zip(lines, low):
            if l in _SIDEBAR_LABELS or "linkedin.com" in l or l.startswith("http"):
                continue
            if 2 <= len(ln) <= 90:
                headline = ln
                break
    return name, headline, parsed_loc


def _seniority_from_text(text):
    """(level, known). 'known' is False only when no seniority signal is present at
    all (so the caller can decline to hard-filter on a mere default guess)."""
    best = None
    for rx, lvl in SENIORITY:
        if rx.search(text or ""):
            best = lvl if best is None else max(best, lvl)
    if best is not None:
        return best, True
    if _ENTRY_RX.search(text or ""):
        return 1, True  # aspiring / new-grad / seeking -> junior, confidently
    return MID, False  # no signal -> default mid, but flagged low-confidence


def _years_from_free_text(text):
    """Professional span from employment date-ranges, EXCLUDING education years.

    The parquet `resume_years` spans every 4-digit year, which on a real resume
    reaches back to high-school/college start years and wildly inflates experience
    (e.g. a 2020 grad reads as 10 yrs). Here we only count month-anchored years
    (employment ranges like 'March 2021'), drop the Education section, and skip years
    sitting next to a degree keyword.
    """
    t = str(text or "")
    if not t:
        return None
    edu = re.search(r"\bEducation\b", t)
    scan = t[: edu.start()] if edu else t
    yrs = []
    for m in _MONTH_YEAR.finditer(scan):
        window = scan[max(0, m.start() - 70) : m.end() + 10]
        if _EDU_KW.search(window):
            continue
        yrs.append(int(m.group(1)))
    if re.search(r"\b(present|current|now)\b", scan, re.I):
        yrs.append(2025)
    if len(yrs) < 2:
        return None
    span = max(yrs) - min(yrs)
    return float(span) if 0 < span <= 45 else None


def _experience_section(text):
    """The work-history block: text between an Experience-ish heading and the next
    section (Education). LinkedIn lists roles reverse-chronologically here."""
    m = re.search(
        r"\b(experience|employment|work history|expériences?|expérience professionnelle|"
        r"parcours(?: professionnel)?|emplois?)\b",
        text or "",
        re.I,
    )
    if not m:
        return ""
    rest = text[m.end() :]
    e = re.search(
        r"\b(education|certifications|skills|volunteer|formation|diplômes?|"
        r"compétences|bénévolat)\b",
        rest,
        re.I,
    )
    return rest[: e.start()] if e else rest


def _recent_title(exp_section):
    """Most-recent role title: in the LinkedIn 'Company / Title / Dates' layout the
    title is the line immediately above the first date range."""
    lines = [ln.strip() for ln in exp_section.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if _MONTH_YEAR.search(ln) and i >= 1:
            return lines[i - 1]
    return ""


def role_titles(text):
    """All role-title lines (most-recent first) from the Experience section. Same
    'title line sits directly above a date range' heuristic as _recent_title, but
    returns every distinct title rather than only the latest — used to seed suggested
    searches. De-duplicated case-insensitively, order preserved."""
    exp = _experience_section(text or "")
    lines = [ln.strip() for ln in exp.split("\n") if ln.strip()]
    out, seen = [], set()
    for i, ln in enumerate(lines):
        if _MONTH_YEAR.search(ln) and i >= 1:
            t = lines[i - 1]
            k = t.lower()
            if k not in seen and 2 <= len(t) <= 80:
                seen.add(k)
                out.append(t)
    return out


# --- employer extraction (for self-employer suppression in personalized results) ---
# A "jobs for you" feed should not surface the seeker's own current/recent employer, so
# we parse the company names out of the profile and let the serving layer drop them.
_DURATION_RX = re.compile(r"^\s*\d+\s+(?:year|yr|month|mo)s?(?:\s+\d+\s+(?:month|mo)s?)?\s*$", re.I)
_BULLET_RX = re.compile(r"^\s*[•·–—*\-]")
_HEADLINE_AT_RX = re.compile(r"\b(?:at|@)\s+(.+)$", re.I)
# common corporate suffixes stripped before comparing two employer names
_EMP_SUFFIX_RX = re.compile(
    r"\b(?:inc|incorporated|llc|ltd|limited|corp|corporation|co|company|gmbh|plc|ag|sa|"
    r"srl|bv|pty|group|holdings|technologies|technology|labs|systems|solutions)\b",
    re.I,
)


def norm_employer(s):
    """Normalize an employer name for comparison: lowercase, drop punctuation and common
    corporate suffixes (Inc/LLC/Ltd/Co/...), collapse whitespace."""
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    s = _EMP_SUFFIX_RX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def same_employer(a, b):
    """True if two employer names denote the same company. Single-token names must match
    exactly (so 'Square' does not match 'Times Square'); a multi-word resume name may
    match as a whole-word substring of the job's employer ('Berkeley Payments' in
    'Berkeley Payments Inc.')."""
    na, nb = norm_employer(a), norm_employer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    if " " not in short:  # single-token name -> require an exact normalized match
        return False
    return re.search(r"\b" + re.escape(short) + r"\b", long) is not None


def _employer_from_headline(headline):
    """Current employer from a LinkedIn-style headline ('Staff PM at Algolia')."""
    m = _HEADLINE_AT_RX.search(headline or "")
    if not m:
        return ""
    # the tail can carry trailing clauses ('at Algolia | building search'); keep the first
    cand = re.split(r"[|•·–—/,]", m.group(1))[0].strip()
    return cand if 2 <= len(cand) <= 60 else ""


def employers(text, headline=""):
    """Recent employer names (most-recent first) for self-employer suppression.

    Reads the LinkedIn 'Experience' block, whose per-role layout is
        Company / [duration] / Title / Month Year - ... / [Location]
    (consecutive roles at one company repeat Title/Dates under a single Company line).
    For each date-range line the title is the line directly above it and the company the
    line above that, skipping a 'N years M months' duration line; when that company slot
    is the previous role's date/location line it's a promotion within the same company and
    is attributed to the company already seen. The headline's '... at <Company>' (the
    current employer) is prepended. De-duplicated case-insensitively, order preserved."""
    out, seen = [], set()

    def add(name):
        name = (name or "").strip()
        key = norm_employer(name)
        if name and key and key not in seen and 2 <= len(name) <= 60:
            seen.add(key)
            out.append(name)

    add(_employer_from_headline(headline))
    exp = _experience_section(text or "")
    lines = [ln.strip() for ln in exp.split("\n") if ln.strip()]
    date_idx = {i for i, ln in enumerate(lines) if _MONTH_YEAR.search(ln)}
    loc_idx = {d + 1 for d in date_idx}  # the line after a date range is that role's location
    for d in sorted(date_idx):
        c = d - 2  # company slot: the line above the title (which sits above the date)
        if c >= 0 and _DURATION_RX.match(lines[c]):
            c -= 1
        if c < 0:
            continue
        if c in date_idx or c in loc_idx:
            continue  # another role under a company already named above
        cand = lines[c]
        if _BULLET_RX.match(cand) or _MONTH_YEAR.search(cand) or _DURATION_RX.match(cand):
            continue
        add(cand)
    return out


# Recent-"role" lines that carry no domain signal — a sabbatical / self-directed gig.
# When the most-recent title is one of these, leading with it (as the embedded text's
# emphasis) drags the profile vector toward generic "building things" prose, so we skip
# to the most-recent CONCRETE role instead.
_PLACEHOLDER_TITLE_RX = re.compile(
    r"\b(independent|freelance|self.?employed|self.?employment|sabbatical|"
    r"career break|between roles|open to work|seeking|tbd|portfolio|personal project)\b",
    re.I,
)


def _corroborated_headline(headline, exp):
    """Headline tokens that actually appear in the work history — so an accurate
    specialization summary ('Search | AI | ML' for someone whose roles are full of
    search) is embedded, while an aspirational headline whose terms never show up in the
    experience (the 'Aspiring AI/ML Engineer' with a sysadmin history) is dropped."""
    if not headline:
        return ""
    low = exp.lower()
    toks = re.findall(r"[A-Za-z][A-Za-z+/.#&-]+", headline)
    kept = [t for t in toks if re.search(r"\b" + re.escape(t.lower()) + r"\b", low)]
    return " ".join(kept)


def query_text(text):
    """Text to embed for matching. Leans on DEMONSTRATED experience rather than a
    self-declared 'Top Skills' sidebar, but LEADS the embedded text with two dense,
    low-boilerplate signals so the profile vector isn't a washed-out centroid of the
    whole document: (1) the headline, but ONLY the part corroborated by the work history
    (drops aspiration, keeps an accurate specialization summary); (2) the most-recent
    CONCRETE role title, skipping placeholder gigs ('Independent Work', a sabbatical)
    that would otherwise dominate with domain-free prose. Falls back to the full blob
    when no Experience section is parseable."""
    text = (text or "").strip()
    exp = _experience_section(text).strip()
    if not exp:
        return text
    lead = []
    sig = _corroborated_headline(_parse_identity(text)[1], exp)
    if sig:
        lead.append(sig)
    rt = _recent_title(exp)
    concrete = rt if (rt and not _PLACEHOLDER_TITLE_RX.search(rt)) else ""
    if not concrete:
        for t in role_titles(text):
            if t and not _PLACEHOLDER_TITLE_RX.search(t):
                concrete = t
                break
    if concrete:
        lead.append(concrete)
    prefix = ". ".join(lead)
    return f"{prefix}. {exp}" if prefix else exp


def specialization_text(text):
    """A short, dense DOMAIN signal: corroborated headline + concrete role titles, with
    NO experience prose. Embedded as a second profile vector for max-sim matching, so a
    specialist (e.g. search) whose long experience centroid washes out to "generic
    senior engineer" still scores high against on-specialty roles. Returns "" when there
    is no distinctive title/headline signal (then the caller skips the second vector)."""
    text = (text or "").strip()
    exp = _experience_section(text)
    head = _corroborated_headline(_parse_identity(text)[1], exp) if exp else ""
    titles = [t for t in role_titles(text) if t and not _PLACEHOLDER_TITLE_RX.search(t)]
    bits = [b for b in ([head] + titles[:6]) if b]
    return ". ".join(bits)


def features_from_text(text, loc=""):
    """Build a feature dict from free-form profile text (paste / .txt / LinkedIn PDF).

    Unlike resume_features (which reads clean structured parquet columns) there is only
    an unstructured blob, so identity/seniority/years use the LinkedIn-aware helpers
    above; degree/creds reuse the shared regex extractors. `seniority_known=False`
    tells axis_status not to hard-filter seniority when it was only a default guess.
    """
    text = (text or "").strip()
    loc = (loc or "").strip()
    name, headline, parsed_loc = _parse_identity(text)
    loc_eff = loc or parsed_loc
    sen, sen_known = _seniority_from_text(text)
    # clean tokens when we have a real location; else permissive whole-text scan
    loc_tok = geo_tokens(loc_eff) if loc_eff else geo_tokens(text)
    return {
        "name": name or "(your profile)",
        "headline": headline[:160],
        "field": profile_field(text, role_titles(text), headline),
        "loc": loc_eff,
        "loc_tok": sorted(loc_tok),
        "country": loc_country(loc_eff),
        "text": text[:2000],
        "seniority": sen,
        "seniority_known": sen_known,
        "years": _years_from_free_text(text),
        "degree": resume_degree(text),
        "creds": sorted(resume_creds(text, "", text)),
        "employers": employers(text, headline),
    }


def job_features(d):
    """Build a feature dict from a parsed metadata.jsonl record."""
    txt = d.get("text") or ""
    locs = job_locations(d)
    loc_tok = set().union(*[geo_tokens(str(x)) for x in locs]) if locs else set()
    # countries the posting offers as a WHOLE-country location (e.g. a 'UK' entry in a
    # multi-location list) — lets a same-country seeker pass the location axis even with
    # no city overlap. City-only entries ('London') contribute nothing here, so two
    # different cities in one country still require token overlap as before.
    loc_countries = sorted({loc_country(str(x)) for x in locs if is_country_only(str(x))})
    return {
        "title": d.get("title", ""),
        "role_family": d.get("role_family") or "other",
        "sen": seniority_of(d.get("title", "")),
        "remote": job_is_remote(d),
        "loc": "; ".join(str(x) for x in locs),
        "loc_tok": sorted(loc_tok),
        "loc_countries": loc_countries,
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


# common metro abbreviations that don't share tokens with their spelled-out form;
# expanded on both sides at compare time so e.g. a "NYC" job matches a "New York City"
# resume. (3+ char only — 2-char abbrevs like LA/SF/DC are dropped by geo_tokens.)
_GEO_SYNONYMS = {
    "nyc": {"new", "york", "city"},
    "philly": {"philadelphia"},
    "vegas": {"las", "vegas"},
    "nola": {"new", "orleans"},
    "atx": {"austin"},
    "bayarea": {"san", "francisco", "bay"},
    "socal": {"los", "angeles", "san", "diego"},
    "norcal": {"san", "francisco"},
}


def _expand_geo(tokens):
    out = set(tokens)
    for t in tokens:
        if t in _GEO_SYNONYMS:
            out |= _GEO_SYNONYMS[t]
    return out


# --- country-level location fallback ---
# Country tokens are too coarse for the normal city/region token overlap (every US city
# would "match" every other on "states"), so geo_tokens drops them via STOP_GEO. But that
# leaves a posting located only at a country ("UK", "United States", or one country entry
# in a multi-location list) with no tokens to overlap, so it can never pass the location
# axis even for someone plainly in that country. These helpers recover that case WITHOUT
# loosening city-level matching: a country-granular location matches a same-country seeker.
# Single-token aliases are kept unambiguous (no 2-letter forms like CA/IN/IT that collide
# with US states / English words); multi-word names are matched as substrings.
_COUNTRY_WORD = {
    "uk": "gb",
    "usa": "us",
    "us": "us",
    "britain": "gb",
    "england": "gb",
    "scotland": "gb",
    "wales": "gb",
    "kingdom": "gb",
    "america": "us",
    "canada": "ca",
    "germany": "de",
    "deutschland": "de",
    "france": "fr",
    "spain": "es",
    "espana": "es",
    "italy": "it",
    "italia": "it",
    "netherlands": "nl",
    "ireland": "ie",
    "india": "in",
    "australia": "au",
    "poland": "pl",
    "portugal": "pt",
    "brazil": "br",
    "mexico": "mx",
    "japan": "jp",
    "singapore": "sg",
    "switzerland": "ch",
    "sweden": "se",
    "norway": "no",
    "denmark": "dk",
    "finland": "fi",
    "belgium": "be",
    "austria": "at",
}
_COUNTRY_PHRASES = [
    ("united kingdom", "gb"),
    ("great britain", "gb"),
    ("u.k", "gb"),
    ("united states", "us"),
    ("u.s.a", "us"),
    ("u.s", "us"),
    ("new zealand", "nz"),
    ("south africa", "za"),
    ("united arab emirates", "ae"),
    ("hong kong", "hk"),
    ("czech republic", "cz"),
    ("south korea", "kr"),
]
# country words to subtract when deciding whether a location is country-granular
_COUNTRY_TOKENS = set(_COUNTRY_WORD) | {w for phrase, _ in _COUNTRY_PHRASES for w in phrase.split()}


def loc_country(s):
    """Canonical 2-letter country code a location string denotes, or '' if none."""
    t = (s or "").lower()
    for phrase, code in _COUNTRY_PHRASES:
        if phrase in t:
            return code
    for tok in re.findall(r"[a-z]+", t):
        if tok in _COUNTRY_WORD:
            return _COUNTRY_WORD[tok]
    return ""


def is_country_only(s):
    """True if a location names ONLY a country (no city/region): it resolves to a country
    and, once country words are removed, has no remaining geo tokens. So 'UK' / 'United
    Kingdom' are country-only but 'London, UK' is not."""
    return bool(loc_country(s)) and not (geo_tokens(s) - _COUNTRY_TOKENS)


# --- metro gazetteer (location-axis fallback) -------------------------------
# Major commute markets: a seeker and a job in the same metro should pass the location
# axis even with no shared geo token. Matched as case-insensitive substrings of the raw
# location string (so "San Jose, CA" and "Bay Area" both resolve to sf_bay).
_METRO_CITIES = {
    "sf_bay": [
        "san francisco",
        "bay area",
        "silicon valley",
        "oakland",
        "san jose",
        "mountain view",
        "palo alto",
        "menlo park",
        "sunnyvale",
        "santa clara",
        "cupertino",
        "redwood city",
        "san mateo",
        "berkeley",
        "fremont",
        "foster city",
        "emeryville",
        "south san francisco",
        "burlingame",
    ],
    "nyc": ["new york", "brooklyn", "manhattan", "queens", "jersey city", "newark", "hoboken"],
    "la": [
        "los angeles",
        "santa monica",
        "pasadena",
        "burbank",
        "culver city",
        "el segundo",
        "long beach",
        "irvine",
        "santa ana",
        "anaheim",
    ],
    "seattle": ["seattle", "bellevue", "redmond", "kirkland", "tacoma", "everett"],
    "boston": ["boston", "cambridge", "somerville", "waltham", "quincy", "newton"],
    "dc": [
        "washington, d",
        "washington dc",
        "arlington",
        "alexandria",
        "bethesda",
        "reston",
        "mclean",
    ],
    "austin": ["austin", "round rock"],
    "chicago": ["chicago", "evanston", "naperville"],
    "denver": ["denver", "boulder", "aurora"],
    "atlanta": ["atlanta", "alpharetta", "marietta"],
    "london": ["london"],
    "paris": ["paris"],
}


def metros_of(s):
    """Set of major metros a raw location string resolves to (substring match)."""
    s = (s or "").lower()
    return {m for m, cities in _METRO_CITIES.items() if any(c in s for c in cities)}


# --- field / role-family alignment axis -------------------------------------
# Coarse professional FIELD per index role_family. The field axis blocks cross-field
# drift (a software/ML engineer should not be filtered INTO marketing / recruiting /
# sales roles) while allowing in-field drift (search -> other software/ML/AI/data
# engineering), which is expected and desirable. We map both the resume and the job to
# a coarse field and check compatibility.
TECH_FAMILIES = frozenset(
    {
        "software_engineering",
        "data_engineering",
        "data_science_ml",
        "data_analytics",
        "ai_ml",
        "ai_data_annotation",
        "devops_sre_infra",
        "security",
        "research_academic",
    }
)
_FAMILY_FIELD = {
    **{f: "tech" for f in TECH_FAMILIES},
    "product_management": "product",
    "project_program_management": "product",
    "design_ux": "design",
    "marketing": "marketing",
    "sales": "sales",
    "customer_success_support": "cs",
    "operations_admin": "ops",
    "finance_accounting": "finance",
    "legal": "legal",
    "hr_people_ops": "hr",
    "healthcare_clinical": "healthcare",
    "healthcare_allied": "healthcare",
    "healthcare_admin": "healthcare",
    "education_teaching": "education",
    "skilled_trades_construction": "trades",
    "transportation_logistics": "logistics",
    "food_service_hospitality": "food",
    "retail": "retail",
    "creative_content": "creative",
    "manufacturing_production": "manufacturing",
    "public_safety": "safety",
    "nonprofit_social_services": "nonprofit",
    "consulting_strategy": "consulting",
}
# Job fields considered in-field for a given PROFILE field. A field maps to itself plus
# genuinely adjacent fields; everything else is cross-field and blocked under the
# qualify-filter. Unknown job field ("other") is never hard-dropped (low-confidence).
_FIELD_COMPAT = {
    "tech": {"tech", "product", "design"},
    "product": {"product", "tech", "design"},
    "design": {"design", "product", "tech", "creative"},
    "marketing": {"marketing", "sales", "creative", "product"},
    "sales": {"sales", "marketing", "cs"},
    "cs": {"cs", "sales", "ops"},
    "finance": {"finance", "consulting", "ops"},
    "hr": {"hr", "ops"},
    "ops": {"ops", "logistics", "hr"},
    "healthcare": {"healthcare"},
    "education": {"education"},
    "trades": {"trades", "manufacturing", "logistics"},
    "logistics": {"logistics", "trades", "ops"},
    "food": {"food", "retail"},
    "retail": {"retail", "sales", "food"},
    "creative": {"creative", "marketing", "design"},
    "manufacturing": {"manufacturing", "trades", "logistics"},
    "consulting": {"consulting", "finance", "product", "tech"},
}
# Field keyword cues, scored over a title/headline blob (argmax wins). TECH is listed
# first so it wins ties for hybrid "ML Engineer / Data Scientist"-type phrasings.
_FIELD_CUES = [
    (
        "tech",
        re.compile(
            r"software|engineer|developer|programmer|architect|devops|\bsre\b|machine learning|\bml\b|\bai\b|data scien|research scientist|backend|front.?end|full.?stack|infrastructure|platform",
            re.I,
        ),
    ),
    (
        "product",
        re.compile(
            r"product manager|product owner|product management|program manager|\bgroup pm\b", re.I
        ),
    ),
    ("design", re.compile(r"designer|\bux\b|\bui\b|user experience|user research", re.I)),
    ("marketing", re.compile(r"marketing|growth|\bseo\b|content strateg|brand|demand gen", re.I)),
    (
        "sales",
        re.compile(
            r"\bsales\b|account executive|business development|\bgtm\b|go.to.market|revenue|quota",
            re.I,
        ),
    ),
    ("hr", re.compile(r"recruit|talent|people ops|human resources|\bhr\b|sourcer", re.I)),
    (
        "finance",
        re.compile(r"finance|accountant|accounting|controller|fp&a|bookkeep|treasury", re.I),
    ),
    ("cs", re.compile(r"customer success|customer support|account manager|client services", re.I)),
    ("healthcare", re.compile(r"\bnurse\b|physician|clinical|therapist|\brn\b|caregiver", re.I)),
]


def field_of_text(blob):
    """Coarse field for a title/headline blob (argmax over keyword cues), or "" if no
    cue fires. Title-driven by design: a job description is often company boilerplate
    (an AI startup's recruiter posting reads like an AI-eng role), so the TITLE is the
    reliable field signal."""
    blob = blob or ""
    best, best_n = "", 0
    for field, rx in _FIELD_CUES:
        n = len(rx.findall(blob))
        if n > best_n:
            best, best_n = field, n
    return best


def profile_field(text, titles=None, headline=""):
    """Best-effort coarse field for a resume. Scores role titles + headline (densest,
    least-boilerplate signal) AND the resume body, by argmax over cue counts — so a
    profile whose titles parse poorly (year-only dates, odd layout) still classifies
    from its prose. A resume is the seeker's own text, not company boilerplate, so the
    body is a safe signal here. "" (unknown) makes the field axis a no-op."""
    blob = " ".join((titles or []) + [headline or "", (text or "")[:1500]])
    return field_of_text(blob)


def job_field(j):
    """Coarse field for a job: its role_family when confidently classified, else the
    title (cuts through boilerplate-heavy descriptions). "" when unknown."""
    rf = j.get("role_family") or ""
    if rf and rf != "other" and rf in _FAMILY_FIELD:
        return _FAMILY_FIELD[rf]
    return field_of_text(j.get("title") or "")


def axis_status(r, j):
    """Return per-axis pass/fail + human-readable reasons for a (resume, job) pair.

    r and j are feature dicts (loc_tok/creds/cred_gates are lists here).
    """
    r_loc = _expand_geo(set(r["loc_tok"]))
    j_loc = _expand_geo(set(j["loc_tok"]))
    r_creds = set(r["creds"])
    j_gates = set(j["cred_gates"])

    # seniority — hard-filter ONLY under-qualification (a junior genuinely can't fill a
    # senior role). Over-qualification is mostly title inflation plus a question of the
    # candidate's interest, neither of which is readable from a resume: a "Head of" at a
    # 1-person shop may be a director / first-line manager at a larger company. So let it
    # pass, lean on cosine, and flag it as a soft note only. Also skip a mere default
    # guess (free-text resume with no seniority signal).
    sen_known = r.get("seniority_known", True)
    under_gap = j["sen"] - r["seniority"]  # > 0 means the candidate is BELOW the job
    sen_ok = (not sen_known) or under_gap < 2
    if not sen_ok:
        sen_reason = (
            f"underqualified ({SENIORITY_LABELS[r['seniority']]} vs {SENIORITY_LABELS[j['sen']]})"
        )
    elif sen_known and r["seniority"] - j["sen"] >= 2:
        sen_reason = f"may be overqualified ({SENIORITY_LABELS[r['seniority']]} vs {SENIORITY_LABELS[j['sen']]})"
    else:
        sen_reason = ""

    # location — city/region token overlap, OR remote, OR (country-level fallback) the
    # posting lists the seeker's whole country as one of its locations, which a bare-country
    # posting ('UK') otherwise can't express in city tokens.
    loc_ok = j["remote"] or bool(r_loc & j_loc)
    if not loc_ok and r.get("country") and r["country"] in set(j.get("loc_countries", [])):
        loc_ok = True
    # metro fallback: a seeker in "San Francisco Bay Area" and a job in "Mountain View"
    # share no geo TOKENS but are the same commute market. Pass when both resolve to the
    # same major metro (cuts false location rejections of on-site, same-metro roles).
    if not loc_ok:
        rm, jm = metros_of(r.get("loc", "")), metros_of(j.get("loc", ""))
        if rm & jm:
            loc_ok = True
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

    # field — block cross-FIELD drift (don't route a tech profile INTO marketing /
    # recruiting / sales). Enforced only when BOTH the profile field and the job field
    # are known; in-field + adjacent drift stays allowed (_FIELD_COMPAT).
    r_field = r.get("field") or ""
    j_field = job_field(j)
    if r_field and j_field and r_field in _FIELD_COMPAT:
        field_ok = j_field in _FIELD_COMPAT[r_field]
    else:
        field_ok = True
    field_reason = "" if field_ok else f"different field ({j_field} vs your {r_field} focus)"

    return {
        "sen": {"ok": sen_ok, "reason": sen_reason},
        "loc": {"ok": loc_ok, "reason": loc_reason},
        "gate": {"ok": gate_ok, "reason": "; ".join(gate_reasons)},
        "field": {"ok": field_ok, "reason": field_reason},
        "all": sen_ok and loc_ok and gate_ok and field_ok,
    }
