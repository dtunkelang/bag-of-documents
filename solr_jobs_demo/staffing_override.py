"""Per-job industry override for staffing/employment-agency employers.

The slug-level industry label is wrong for staffing agencies, whose listings
span many real industries (a nurse posted by rightworks should facet under
healthcare_provider, not tech_software_internet or even consulting_professional_services).

Strategy:
  - Detect staffing agencies by slug pattern + curated known names.
  - For those, derive industry from the job's role_family (already produced by
    solr_jobs_demo/facets/heuristics.py).
  - Fall back to consulting_professional_services when role_family is too
    generic to imply an industry (sales, hr_people_ops, etc.) — that's the
    taxonomy-documented default for "industry not discernible from job".
"""

from __future__ import annotations

import re

# Staffing pattern: tokens that almost always indicate an agency.
# "consulting" is intentionally excluded — real engineering/strategy firms
# (cha-consulting-inc, deloitte-consulting-sea) use it without being agencies.
_STAFFING_TOKEN = re.compile(
    r"(?:^|-)(staffing|recruit|recruiter|recruiters|recruiting|"
    r"talent|placement|workforce|manpower|temps?|tempstaff)(?:-|$)"
)

# Curated extras: major agencies whose slug doesn't contain a staffing token.
_KNOWN_AGENCY_SLUGS = frozenset(
    {
        "rightworks",
        "tandym-group",
        "robert-half",
        "randstad-usa",
        "randstad-digital-americas",
        "kforce",
        "aerotek",
        "adecco",
        "kelly-services",
        "actalent",
        "insight-global",
        "tek-systems",
        "teksystems",
        "motion-recruitment",
        "axiomtalentplatform",
        "talentify-io",
        "beacon-hill-staffing-group",
        "complete-staffing-solutions",
        "ultimate-staffing",
        "creative-financial-staffing-cfs",
        "talentburst-an-inc-5000-company",
        "talent-groups",
        "the-global-talent-co",
        "24-seven-talent",
        "growetalents",
        "hiretalent-diversity-staffing-recruiting-firm",
        "millennium-recruiting-inc",
        "goodwin-recruiting",
        "twiceasnice-recruiting",
        "buyersedgeplatformrecruiting",
        "print-recruiting",
        "crisprecruit",
        "team-builders-recruiting-and-consulting-llc",
        "aditi-consulting",
        "pyramid-consulting-inc",
        "manpowergroup",
        "inter-island-manpower-pte-ltd",
        "persolkelly-workforce-solutions-malaysia-sdn-bhd",
        "healthtrust-workforce-solutions",
        "maxim-healthcare-staffing",
        "medpro-healthcare-staffing",
        "national-staffing-solutions",
        "protouch-staffing",
        "bluebird-staffing",
        "agensi-pekerjaan-js-staffing-services-sdn-bhd",
        "agensi-pekerjaan-achieve-career-consultant-m-sdn-bhd-jtksm-579",
        "agensi-pekerjaan-randstad-sdn-bhd-professional",
        "agensi-pekerjaan-gmrecruitment-sdn-bhd",
    }
)

# Malaysian/SG/HK agency prefix — "agensi-pekerjaan" literally means "job agency"
_AGENSI_PREFIX = re.compile(r"^agensi-pekerjaan(-|$)")


def is_staffing_agency(slug: str) -> bool:
    if not slug:
        return False
    if slug in _KNOWN_AGENCY_SLUGS:
        return True
    if _AGENSI_PREFIX.match(slug):
        return True
    return bool(_STAFFING_TOKEN.search(slug))


# role_family -> industry. Empty string means "no override, use fallback".
# Function-only roles (sales, marketing, hr, finance_accounting, ops, etc.)
# don't tell us the hiring industry and stay empty.
ROLE_TO_INDUSTRY: dict[str, str] = {
    "software_engineering": "tech_software_internet",
    "data_engineering": "tech_software_internet",
    "data_science_ml": "tech_software_internet",
    "devops_sre_infra": "tech_software_internet",
    "product_management": "tech_software_internet",
    "design_ux": "tech_software_internet",
    "security": "tech_software_internet",
    "healthcare_clinical": "healthcare_provider",
    "healthcare_allied": "healthcare_provider",
    "healthcare_admin": "healthcare_provider",
    "legal": "legal_services",
    "skilled_trades_construction": "real_estate_construction",
    "manufacturing_production": "manufacturing",
    "transportation_logistics": "transportation_logistics",
    "education_teaching": "education_higher",
    "research_academic": "education_higher",
    "public_safety": "public_sector_government",
    "food_service_hospitality": "hospitality_food_service",
    "retail": "retail_ecommerce",
    "nonprofit_social_services": "nonprofit",
    # left empty (function spans industries): sales, marketing, hr_people_ops,
    # operations_admin, customer_success_support, creative_content,
    # consulting_strategy, project_program_management, finance_accounting,
    # engineer (generic), other
}

FALLBACK_INDUSTRY = "consulting_professional_services"

# Slug-industry values we treat as "generic default" — these are usually the
# label assigned when the LLM/propagator couldn't tell. When a staffing slug
# carries one of these, we trust role_family over the slug.
# A staffing slug with a specialty industry (healthcare_provider, legal_services,
# finance_banking, etc.) is treated as a vertical agency and kept as-is unless
# the role_family is a strong cross-industry signal (see _STRONG_ROLE_OVERRIDE).
_GENERIC_SLUG_INDUSTRIES = frozenset(
    {
        "",
        "unclassified",
        "consulting_professional_services",
        "tech_software_internet",
    }
)

# Roles strong enough to override even a specialty staffing slug: the role
# tells us the industry more reliably than the agency's claimed specialty.
_STRONG_ROLE_OVERRIDE = frozenset(
    {
        "software_engineering",
        "data_engineering",
        "data_science_ml",
        "devops_sre_infra",
        "security",
        "legal",
        "healthcare_clinical",
        "healthcare_allied",
    }
)


# Title-keyword fallbacks for staffing jobs whose role_family came back too
# generic (heuristics.py misses some specialist titles). Patterns are narrow on
# purpose: only words that uniquely imply an industry.
_TITLE_INDUSTRY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(physician|surgeon|psychiatrist|psychologist|neurologist|"
            r"dermatologist|pediatric|obgyn|ob/gyn|cardiologist|anesthesiologist|"
            r"radiologist|oncologist|gastroenterologist|pathologist|urologist|"
            r"orthopedic|nephrologist|hematologist|endocrinologist|"
            r"ophthalmologist|otolaryngologist|hospitalist|"
            r"family medicine|internal medicine|primary care|"
            r"nurse practitioner|physician assistant|midwife|sonographer|"
            r"licensed practical|registered nurse|speech[- ]?language pathologist|"
            r"speech pathologist|occupational therap|physical therap|respiratory therap|"
            r"dentist|orthodontist|veterinarian|paramedic|pharmacist|"
            r"phlebotomist|radiographer|medical assistant)\b",
            re.I,
        ),
        "healthcare_provider",
    ),
    (
        re.compile(r"\b(attorney|paralegal|litigation associate|patent agent)\b", re.I),
        "legal_services",
    ),
    (
        re.compile(r"\b(teacher|professor|instructor|adjunct|tutor|lecturer)\b", re.I),
        "education_higher",
    ),
]


def _title_industry(title: str) -> str:
    if not title:
        return ""
    for pat, ind in _TITLE_INDUSTRY_PATTERNS:
        if pat.search(title):
            return ind
    return ""


def resolve_industry(slug: str, slug_industry: str, role_family: str, title: str = "") -> str:
    """Return the industry to file this job under.

    - Non-staffing employer: use slug_industry as-is.
    - Staffing employer:
        * If slug-industry is generic (or missing), prefer role-derived
          industry, then title-keyword fallback, then
          consulting_professional_services.
        * If slug-industry is a specialty (healthcare_provider, etc.),
          keep it unless role_family is a strong cross-industry signal,
          or the title clearly implies a different industry.
    """
    if not is_staffing_agency(slug):
        return slug_industry or "unclassified"

    role = role_family or ""
    title_role_industry = ROLE_TO_INDUSTRY.get(role, "")
    title_kw_industry = _title_industry(title)

    if slug_industry in _GENERIC_SLUG_INDUSTRIES:
        return title_role_industry or title_kw_industry or FALLBACK_INDUSTRY

    # Specialty staffing slug: only override on strong role signal or
    # unambiguous title keyword.
    if role in _STRONG_ROLE_OVERRIDE and title_role_industry:
        return title_role_industry
    if title_kw_industry and title_kw_industry != slug_industry:
        return title_kw_industry
    return slug_industry
