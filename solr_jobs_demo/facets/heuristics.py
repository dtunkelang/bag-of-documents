"""Heuristic facet extractor for the jobs corpus.

Quality is meaningfully below an LLM but ships for free. Trades cost for accuracy.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from taxonomy import TECH_STACK

# ===== role_family =====


# Title-keyword patterns. First match wins, so order matters: most specific first.
# Each tuple is (compiled regex, role_family enum value).
def _ci(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


ROLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # healthcare_clinical — high specificity to win against generic "engineer"
    (
        _ci(
            r"\b(registered nurse|RN|LPN|LVN|CNA|nurse practitioner|NP|"
            r"physician|MD\b|doctor|surgeon|pharmacist|dentist|veterinarian|"
            r"psychiatrist|psychologist|therapist|physical therapist|PT\b|"
            r"occupational therapist|OT\b|respiratory therapist|"
            r"nursing assistant|nurse aide|midwife|EMT|paramedic|"
            r"radiologic technologist|sonographer|phlebotomist)\b"
        ),
        "healthcare_clinical",
    ),
    (
        _ci(
            r"\b(medical (lab|laboratory) technician|lab tech|MLT\b|"
            r"radiology tech|imaging tech|ultrasound tech|pharmacy tech|"
            r"dental hygienist|dental assistant|medical assistant)\b"
        ),
        "healthcare_allied",
    ),
    (
        _ci(
            r"\b(medical (biller|coder|coding|billing|scheduler|records)|"
            r"hospital administrat|patient services coordinator|patient access)\b"
        ),
        "healthcare_admin",
    ),
    # security
    (
        _ci(
            r"\b(security engineer|security analyst|security architect|"
            r"infosec|cybersecurity|cyber security|penetration tester|pentester|"
            r"SOC analyst|threat (analyst|intel|hunt)|incident response|"
            r"application security|appsec|GRC)\b"
        ),
        "security",
    ),
    # devops / sre / infra / platform
    (
        _ci(
            r"\b(DevOps|SRE\b|site reliability|platform engineer|infrastructure engineer|"
            r"cloud engineer|systems engineer|sysadmin|system administrator|"
            r"network engineer|kubernetes engineer|reliability engineer)\b"
        ),
        "devops_sre_infra",
    ),
    # data engineering — must come before generic data scientist patterns
    (
        _ci(
            r"\b(data engineer|analytics engineer|ETL developer|"
            r"big data engineer|ML platform engineer)\b"
        ),
        "data_engineering",
    ),
    # data science / ML — broad
    (
        _ci(
            r"\b(data scientist|ML engineer|machine learning engineer|"
            r"AI engineer|AI scientist|research scientist|applied scientist|"
            r"MLOps|deep learning|NLP engineer|computer vision engineer|"
            r"data analyst|business analyst|BI analyst|quantitative analyst|quant analyst)\b"
        ),
        "data_science_ml",
    ),
    # software engineering — broad, after specific eng specialties.
    # Also catches "Architect", "Engineering Manager/Director/VP" titles and
    # "Developer"-family standalone roles.
    (
        _ci(
            r"\b(software engineer|software developer|SDE\b|SWE\b|"
            r"backend engineer|back[- ]end|frontend engineer|front[- ]end|"
            r"full[- ]stack|web developer|mobile developer|iOS developer|"
            r"android developer|game developer|firmware engineer|embedded engineer|"
            r"QA engineer|test engineer|SDET|automation engineer|"
            r"engineering (manager|director|lead|head)|head of engineering|"
            r"director of engineering|VP of engineering|"
            r"software (architect|engineering)|solutions architect|"
            r"technical architect|principal architect|enterprise architect|"
            r"cloud architect|systems? architect|"
            r"developer\b|"
            r"engineer\b)\b"
        ),
        "software_engineering",
    ),
    # design
    (
        _ci(
            r"\b(product designer|UX designer|UI designer|UI/UX|"
            r"user experience|user interface|visual designer|interaction designer|"
            r"design (lead|director|manager))\b"
        ),
        "design_ux",
    ),
    # product management
    (
        _ci(
            r"\b(product manager|PM\b|product owner|group product manager|GPM|"
            r"chief product officer|CPO\b)\b"
        ),
        "product_management",
    ),
    # project/program management — must come after "product manager"
    (
        _ci(
            r"\b(project manager|program manager|TPM\b|technical program manager|"
            r"scrum master|delivery manager|project coordinator)\b"
        ),
        "project_program_management",
    ),
    # marketing
    (
        _ci(
            r"\b(marketing|growth|brand|SEO|SEM|content marketing|"
            r"social media manager|community manager|PR\b|public relations|"
            r"copywriter|demand gen|email marketing|product marketing|PMM\b)\b"
        ),
        "marketing",
    ),
    # sales — includes pre-sales, market dev, partnerships, comma-titles
    (
        _ci(
            r"\b(account executive|AE\b|accounts executive|"
            r"sales (rep|representative|development|"
            r"manager|director|engineer|specialist|strategy|consultant|associate)|"
            r"SDR\b|BDR\b|business development|biz dev|"
            r"inside sales|outside sales|sales associate|territory manager|"
            r"market development|partnerships (manager|director)|"
            r"pre[- ]sales|presales|"
            r"VP,?\s+(.+\s+)?sales|director,?\s+(.+\s+)?sales|"
            r"manager,?\s+(.+\s+)?sales|head of sales|"
            r"head of (revenue|growth)|VP (of )?revenue)\b"
        ),
        "sales",
    ),
    # customer success / support / pre-post-sale specialists
    (
        _ci(
            r"\b(customer success|CSM\b|customer support|technical support|"
            r"help desk|customer service rep|customer service|CX\b|"
            r"account manager(?! of)|client services|client success|"
            r"client (& )?partner|implementation engineer|solutions engineer|"
            r"sales engineer|SE\b|"
            r"solutions consultant|technical solutions|technical consultant|"
            r"support specialist|support engineer|support representative|"
            r"customer experience|deal desk|onboarding)\b"
        ),
        "customer_success_support",
    ),
    # finance / accounting — includes audit, equity research, deductions, corp finance
    (
        _ci(
            r"\b(accountant|accounting|accounts payable|accounts receivable|"
            r"AP\b clerk|AR\b clerk|controller|CPA\b|auditor|"
            r"audit (manager|director|associate|senior)|senior audit|"
            r"bookkeeper|"
            r"financial analyst|FP&A|financial planning|treasury|"
            r"tax(?: associate| analyst| director| manager)?|"
            r"investment banking|portfolio manager|underwriter|"
            r"equity research|deductions specialist|"
            r"corporate finance|head of (corporate )?finance|"
            r"banca product|bancassurance|"
            r"chief financial officer|CFO\b)\b"
        ),
        "finance_accounting",
    ),
    # legal
    (
        _ci(
            r"\b(attorney|lawyer|paralegal|legal counsel|compliance officer|"
            r"general counsel|associate attorney|law clerk)\b"
        ),
        "legal",
    ),
    # hr / people ops
    (
        _ci(
            r"\b(recruiter|talent acquisition|TA partner|HRBP|HR\b|"
            r"human resources?|human resource administrator|"
            r"people operations|people ops|"
            r"people partner|people & culture|talent manager|sourcer|HR generalist|"
            r"compensation analyst|benefits|"
            r"learning and development|L&D\b|training specialist|"
            r"performance management|"
            r"chief people officer|CPO\b people|CHRO\b)\b"
        ),
        "hr_people_ops",
    ),
    # education
    (
        _ci(
            r"\b(teacher|tutor|instructor|professor|lecturer|"
            r"adjunct|teaching assistant|TA\b|education coordinator|"
            r"curriculum designer|principal of (?:education|school))\b"
        ),
        "education_teaching",
    ),
    # research / academic
    (
        _ci(
            r"\b(postdoc|postdoctoral|research associate|research scientist|"
            r"PhD candidate|research fellow|lab manager)\b"
        ),
        "research_academic",
    ),
    # skilled trades + construction
    (
        _ci(
            r"\b(electrician|plumber|HVAC|carpenter|welder|machinist|"
            r"mechanic|technician|millwright|pipefitter|sheet metal|"
            r"construction worker|laborer|foreman|superintendent|"
            r"site supervisor|estimator|construction manager)\b"
        ),
        "skilled_trades_construction",
    ),
    # transportation / logistics
    (
        _ci(
            r"\b(driver|truck driver|CDL|delivery driver|courier|"
            r"dispatcher|warehouse|forklift|material handler|"
            r"logistics|supply chain|fleet manager|freight)\b"
        ),
        "transportation_logistics",
    ),
    # food / hospitality
    (
        _ci(
            r"\b(chef|sous chef|line cook|prep cook|baker|bartender|"
            r"server|waiter|waitress|host(?:ess)?|barista|"
            r"hotel|housekeep|concierge|front desk|hospitality|"
            r"restaurant manager|catering)\b"
        ),
        "food_service_hospitality",
    ),
    # retail
    (
        _ci(
            r"\b(cashier|retail|store manager|store associate|"
            r"sales associate|stock associate|merchandiser|"
            r"visual merchandiser|loss prevention)\b"
        ),
        "retail",
    ),
    # creative content
    (
        _ci(
            r"\b(writer|editor|journalist|content creator|copywriter(?! marketing)|"
            r"illustrator|graphic designer|video editor|videographer|"
            r"photographer|animator|art director|creative director)\b"
        ),
        "creative_content",
    ),
    # manufacturing / production
    (
        _ci(
            r"\b(machine operator|production (worker|associate|technician)|"
            r"assembly|assembler|manufacturing engineer|process engineer|"
            r"quality (engineer|inspector|technician)|industrial engineer)\b"
        ),
        "manufacturing_production",
    ),
    # public safety
    (
        _ci(
            r"\b(police officer|deputy sheriff|firefighter|"
            r"corrections officer|security officer|security guard|"
            r"loss prevention officer|park ranger|safety officer)\b"
        ),
        "public_safety",
    ),
    # nonprofit / social services
    (
        _ci(
            r"\b(social worker|case manager|case worker|"
            r"community outreach|youth counselor|advocacy|"
            r"program coordinator(?! engineering)|nonprofit)\b"
        ),
        "nonprofit_social_services",
    ),
    # consulting / strategy
    (
        _ci(
            r"\b(management consultant|strategy consultant|"
            r"associate consultant|partner, consulting|"
            r"director of strategy|head of strategy|strategist|"
            r"strategic (planning|advisor)|chief of staff)\b"
        ),
        "consulting_strategy",
    ),
    # operations / admin — broad fallback before "other"
    (
        _ci(
            r"\b(operations manager|operations analyst|operations associate|"
            r"operations (lead|leader|director|coordinator|specialist)|"
            r"executive assistant|administrative assistant|admin assistant|"
            r"personal assistant|receptionist|secretary|"
            r"office manager|office coordinator|business operations|biz ops|"
            r"fleet (manager|administrator)|"
            r"procurement|sourcing manager|buyer\b|category manager|"
            r"facilities (manager|coordinator)|"
            r"program assistant|program coordinator|"
            r"asset management|portfolio (administration|administrator)|"
            r"grants (manager|administrator)|"
            r"compliance (manager|specialist|officer)|"
            r"deal desk|order management|"
            r"service delivery|"
            r"EHS|environmental health|safety (manager|coordinator|specialist|supervisor))\b"
        ),
        "operations_admin",
    ),
    # broader catches — last-chance specializations
    (
        _ci(
            r"\b(packing operator|machine operator|production operator|"
            r"maintenance operator|line operator|equipment operator|"
            r"plant operator|process operator)\b"
        ),
        "manufacturing_production",
    ),
    (_ci(r"\bconstruction inspector|site inspector\b"), "skilled_trades_construction"),
    (_ci(r"\bstylist\b|\bsalesperson\b"), "retail"),
    (_ci(r"\binterior (designer|design|architecture)\b"), "creative_content"),
    (_ci(r"\b(data annotator|data labeler|labeler)\b"), "data_engineering"),
]


def classify_role_family(title: str) -> str:
    for pat, family in ROLE_PATTERNS:
        if pat.search(title):
            return family
    return "other"


# ===== seniority =====

# Order: most specific first.
SENIORITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        _ci(
            r"\bchief (executive|technology|financial|operating|product|"
            r"information|security|marketing|people|legal|data) officer\b|"
            r"\b(CEO|CTO|CFO|COO|CPO|CIO|CISO|CMO|CHRO|CDO)\b"
        ),
        "c_level",
    ),
    (_ci(r"\bvice president\b|\bSVP\b|\bEVP\b|\bVP\b"), "vp"),
    (_ci(r"\b(senior director|sr\.?\s*director)\b"), "director"),
    (_ci(r"\bdirector\b"), "director"),
    (_ci(r"\b(senior manager|sr\.?\s*manager)\b"), "senior_manager"),
    # 'Senior' anywhere maps to senior — checked BEFORE 'manager' so that
    # "Senior Product Manager", "Senior X Engineer" etc. become senior, not manager.
    # (Adjacent "Senior Manager" was already caught above.)
    (_ci(r"\b(senior|sr\.?)\b"), "senior"),
    (_ci(r"\bmanager\b"), "manager"),
    (
        _ci(
            r"\b(team lead|tech lead|engineering lead|lead (engineer|developer|designer|"
            r"product manager|data scientist|analyst|consultant))\b"
        ),
        "lead",
    ),
    (
        _ci(
            r"\b(staff (engineer|developer|scientist|product manager|designer)|"
            r"principal (engineer|developer|scientist|consultant))\b"
        ),
        "staff",
    ),
    # 'Assistant' / 'Apprentice' titles map to entry (Assistant Teacher, Apprentice Plumber, etc.)
    (
        _ci(
            r"\b(assistant|apprentice|trainee|associate (?!consultant|attorney|director))"
            r"\s+(teacher|guide|cook|chef|stylist|technician|nurse|engineer|developer|"
            r"designer|analyst|coordinator|specialist|representative|operator|"
            r"administrator)\b"
        ),
        "entry",
    ),
    (_ci(r"\b(junior|jr\.?)\b"), "junior"),
    (_ci(r"\b(intern|internship)\b"), "intern"),
    (_ci(r"\bentry[- ]level\b"), "entry"),
    (
        _ci(r"\b(I{1,3})\b"),  # roman-numeral level — caller will demote to junior/mid/senior
        "mid",
    ),
]


_SENIORITY_DEFAULT_ROLES = re.compile(
    r"\b(engineer|developer|analyst|designer|scientist|consultant|"
    r"associate|specialist|coordinator|technician|representative|"
    r"officer|assistant|operator)\b",
    re.IGNORECASE,
)


def classify_seniority(title: str) -> str:
    for pat, level in SENIORITY_PATTERNS:
        if pat.search(title):
            return level
    # If title clearly names a role family but no level word, default to mid
    # rather than "not_specified". This is a softer signal but more useful for
    # a facet view than 60% unknowns.
    if _SENIORITY_DEFAULT_ROLES.search(title):
        return "mid"
    return "not_specified"


# ===== remote_mode =====

REMOTE_TOKENS = re.compile(
    r"\b(fully\s+remote|remote[- ](first|only)|100% remote|work from home|WFH|"
    r"telecommut|tele[- ]?work|tele[- ]?health|tele[- ]?medicine|tele[- ]?mental|"
    r"distributed team|remote\s+(?:US|EU|EMEA|APAC|global)\b)\b",
    re.IGNORECASE,
)
HYBRID_TOKENS = re.compile(
    r"\bhybrid\b|\b(?:[2-4]\s+days?\s+in[- ](office|the office)|"
    r"in[- ]office\s+[2-4]\s+days?)\b",
    re.IGNORECASE,
)
ONSITE_TOKENS = re.compile(
    r"\b(on[- ]site only|in[- ]office only|no remote|must be on[- ]site|"
    r"fully on[- ]site)\b",
    re.IGNORECASE,
)


def classify_remote_mode(locations: list[str], description: str, title: str = "") -> str:
    """Scan locations + title + description. Title carries strong signals
    like 'hybrid work model' or 'tele-mental health' that the locations
    field alone misses."""
    locs_blob = " | ".join(locations or [])
    text = (title or "") + "\n" + (description or "")
    if re.search(r"\bremote\b", locs_blob, re.IGNORECASE) and not HYBRID_TOKENS.search(text):
        return "remote"
    if HYBRID_TOKENS.search(text) or HYBRID_TOKENS.search(locs_blob):
        return "hybrid"
    if REMOTE_TOKENS.search(text) and not HYBRID_TOKENS.search(text):
        return "remote"
    if ONSITE_TOKENS.search(text):
        return "on_site"
    # If there's an explicit physical location and no remote/hybrid language, call it on_site.
    if locations and not re.search(r"\bremote\b", locs_blob, re.IGNORECASE):
        return "on_site"
    return "not_specified"


# ===== location =====

US_STATE_ABBR = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}
US_STATE_NAME = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}
COUNTRY_ALIASES = {
    "united states": "US",
    "usa": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "u.k.": "GB",
    "england": "GB",
    "india": "IN",
    "singapore": "SG",
    "philippines": "PH",
    "canada": "CA",
    "germany": "DE",
    "france": "FR",
    "japan": "JP",
    "australia": "AU",
    "netherlands": "NL",
    "spain": "ES",
    "italy": "IT",
    "brazil": "BR",
    "mexico": "MX",
    "ireland": "IE",
    "poland": "PL",
    "switzerland": "CH",
    "sweden": "SE",
    "denmark": "DK",
    "norway": "NO",
    "finland": "FI",
    "south korea": "KR",
    "korea": "KR",
    "china": "CN",
    "hong kong": "HK",
    "taiwan": "TW",
    "new zealand": "NZ",
    "south africa": "ZA",
    "israel": "IL",
    "united arab emirates": "AE",
    "uae": "AE",
    "saudi arabia": "SA",
    "indonesia": "ID",
    "thailand": "TH",
    "vietnam": "VN",
    "malaysia": "MY",
    "argentina": "AR",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE",
}


def parse_location(locations: list[str]) -> tuple[str, str, str]:
    """Return (country_iso2, state_abbr_or_empty, city_or_empty) from the first
    location string. Country defaults to 'US' for unqualified 'City, ST' strings."""
    if not locations:
        return ("", "", "")
    s = (locations[0] or "").strip()
    if not s or s.lower() == "remote":
        return ("", "", "")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return ("", "", "")
    country = ""
    state = ""
    city = ""
    # Try to identify country from the last part
    last = parts[-1].lower()
    if last in COUNTRY_ALIASES:
        country = COUNTRY_ALIASES[last]
        parts = parts[:-1]
    elif parts[-1].upper() in US_STATE_ABBR or parts[-1].lower() in US_STATE_NAME:
        country = "US"
    elif len(parts) == 1 and parts[0].lower() in COUNTRY_ALIASES:
        # Just a country
        country = COUNTRY_ALIASES[parts[0].lower()]
        return (country, "", "")

    # State (US only) — second-to-last after stripping country
    if country == "US" and parts:
        cand = parts[-1].strip()
        if cand.upper() in US_STATE_ABBR:
            state = cand.upper()
            parts = parts[:-1]
        elif cand.lower() in US_STATE_NAME:
            state = US_STATE_NAME[cand.lower()]
            parts = parts[:-1]
    if parts:
        city = parts[0].strip()
    return (country, state, city)


# ===== posted_bucket =====


def classify_posted_bucket(posted_at: str, now: datetime | None = None) -> str:
    if not posted_at:
        return "older"
    try:
        # Tolerate fractional seconds + timezone
        s = posted_at.rstrip("Z")
        # Strip fractional seconds beyond 6 digits if present
        s = re.sub(r"\.(\d{6})\d+", r".\1", s)
        if "+" not in s and "-" not in s[10:]:
            s += "+00:00"
        dt = datetime.fromisoformat(s)
    except Exception:
        return "older"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = ((now or datetime.now(timezone.utc)) - dt).days
    if age_days < 1:
        return "past_24h"
    if age_days < 7:
        return "past_7d"
    if age_days < 30:
        return "past_30d"
    if age_days < 90:
        return "past_90d"
    return "older"


# ===== salary_band =====

FX_TO_USD = {
    "USD": 1.0,
    "EUR": 1.1,
    "GBP": 1.25,
    "INR": 0.012,
    "SGD": 0.74,
    "PHP": 0.018,
    "AUD": 0.66,
    "CAD": 0.73,
    "JPY": 0.0067,
    "MXN": 0.058,
    "BRL": 0.20,
    "CHF": 1.13,
    "SEK": 0.097,
    "NOK": 0.094,
    "DKK": 0.15,
    "PLN": 0.25,
    "ILS": 0.27,
    "AED": 0.27,
    "ZAR": 0.054,
}


def classify_salary_band(salary_min, salary_max, currency: str | None) -> str:
    if salary_min is None and salary_max is None:
        return "not_specified"
    lo = float(salary_min) if salary_min is not None else float(salary_max)
    hi = float(salary_max) if salary_max is not None else float(salary_min)
    mid = (lo + hi) / 2
    fx = FX_TO_USD.get((currency or "USD").upper(), 1.0)
    # Heuristic: if mid < 200, treat as hourly; otherwise annual.
    if mid < 200:
        mid_annual = mid * fx * 40 * 52
    else:
        mid_annual = mid * fx
    if mid_annual < 50_000:
        return "under_50k"
    if mid_annual < 75_000:
        return "50k_75k"
    if mid_annual < 100_000:
        return "75k_100k"
    if mid_annual < 150_000:
        return "100k_150k"
    if mid_annual < 200_000:
        return "150k_200k"
    if mid_annual < 300_000:
        return "200k_300k"
    return "300k_plus"


# ===== tech_stack =====


# Map vocab token -> regex pattern that matches it in text.
# We need word boundaries and special handling for tokens with punctuation.
def _build_tech_patterns() -> list[tuple[re.Pattern, str]]:
    pats: list[tuple[re.Pattern, str]] = []
    for tok in TECH_STACK:
        # Escape special chars, add word boundaries where sensible.
        if tok in ("C++", "C#", ".NET"):
            pats.append((re.compile(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])"), tok))
        elif tok == "Go":
            # Match only 'Golang' or 'Go' in clear programming context.
            # Bare 'Go' is too noisy (matches retail brand 'Go Outdoors', etc.).
            pats.append(
                (
                    re.compile(
                        r"\bGolang\b|"
                        r"\b(?:in|with|using|written\s+in|coded?\s+in|develop(?:ing|ed|er|ment)?\s+in)\s+Go\b|"
                        r"\bGo\s+(?:language|programming|developer|engineer|microservices|"
                        r"services|backend|code|programs?|routines?|modules?|packages?)\b"
                    ),
                    tok,
                )
            )
        elif tok == "R":
            pats.append((re.compile(r"(?<![A-Za-z0-9_])R(?:[\s,.;\)]|$)"), tok))
        elif tok == "Ruby on Rails":
            pats.append((re.compile(r"\b(Ruby on Rails|RoR)\b"), tok))
        elif tok == "OpenAI API":
            pats.append((re.compile(r"\bOpenAI API\b|\bGPT-4\b"), tok))
        elif tok == "Hugging Face":
            pats.append((re.compile(r"\bHugging\s*Face\b|\btransformers library\b"), tok))
        elif tok == "Next.js":
            pats.append((re.compile(r"\bNext\.?\s*js\b|\bNextJS\b"), tok))
        elif tok == "Node.js":
            pats.append((re.compile(r"\bNode\.?\s*js\b|\bNodeJS\b"), tok))
        else:
            pats.append((re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE), tok))
    return pats


_TECH_PATTERNS = _build_tech_patterns()


def classify_tech_stack(text: str) -> list[str]:
    found = []
    for pat, tok in _TECH_PATTERNS:
        if pat.search(text):
            found.append(tok)
    return found


# ===== aggregate =====


def classify_record(rec: dict, now: datetime | None = None) -> dict:
    title = (rec.get("title") or "").strip()
    desc = (rec.get("description") or "").strip()
    text = title + "\n\n" + desc
    locations = rec.get("locations") or []
    country, state, city = parse_location(locations)
    return {
        "role_family": classify_role_family(title),
        "seniority": classify_seniority(title),
        "remote_mode": classify_remote_mode(locations, desc, title),
        "location_country": country,
        "location_state": state,
        "location_city": city,
        "posted_bucket": classify_posted_bucket(rec.get("posted_at") or "", now=now),
        "salary_band_usd_annual": classify_salary_band(
            rec.get("salary_min"), rec.get("salary_max"), rec.get("salary_currency")
        ),
        "tech_stack": classify_tech_stack(text),
    }
