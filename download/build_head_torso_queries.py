#!/usr/bin/env python3
"""Generate curated head/torso query expansions for the te3 cache.

Outputs queries new (not already in cache) covering categories real users type
that the existing cache misses: bare skills, skill x role, "jobs in <city>",
remote/seniority/intent variants, industry buckets, workstyle modifiers,
compensation, employer-type buckets, and role x skill x location combos.

Writes:
  unified_jobs/head_torso_queries.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Curated lexicons
# -----------------------------------------------------------------------------

SKILLS = [
    # programming languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "rust",
    "go",
    "golang",
    "ruby",
    "swift",
    "kotlin",
    "scala",
    "php",
    "perl",
    "r",
    "matlab",
    "sql",
    "bash",
    "julia",
    "dart",
    "lua",
    "haskell",
    "clojure",
    "erlang",
    "elixir",
    "objective-c",
    "fortran",
    "groovy",
    "vba",
    "shell",
    "powershell",
    "solidity",
    "assembly",
    "cobol",
    # web / frontend
    "react",
    "angular",
    "vue",
    "svelte",
    "next.js",
    "nuxt",
    "ember",
    "jquery",
    "redux",
    "tailwind",
    "tailwindcss",
    "bootstrap",
    "material-ui",
    "mui",
    "chakra ui",
    "html",
    "css",
    "sass",
    "scss",
    "less",
    "webpack",
    "vite",
    "babel",
    "rollup",
    # mobile
    "ios",
    "android",
    "react native",
    "flutter",
    "xamarin",
    "ionic",
    "cordova",
    # backend
    "node",
    "nodejs",
    "node.js",
    "express",
    "express.js",
    "nestjs",
    "django",
    "flask",
    "fastapi",
    "tornado",
    "spring",
    "spring boot",
    "rails",
    "ruby on rails",
    "laravel",
    "symfony",
    ".net",
    ".net core",
    "asp.net",
    # cloud
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "alibaba cloud",
    "oracle cloud",
    "ibm cloud",
    "digital ocean",
    "vercel",
    "netlify",
    "heroku",
    "cloudflare",
    # devops / orchestration
    "docker",
    "kubernetes",
    "k8s",
    "openshift",
    "podman",
    "helm",
    "istio",
    "jenkins",
    "gitlab",
    "github",
    "github actions",
    "circleci",
    "argocd",
    "terraform",
    "ansible",
    "puppet",
    "chef",
    "pulumi",
    # databases
    "mysql",
    "postgres",
    "postgresql",
    "mongodb",
    "redis",
    "dynamodb",
    "cassandra",
    "elasticsearch",
    "snowflake",
    "bigquery",
    "redshift",
    "databricks",
    "oracle",
    "sql server",
    "mariadb",
    "sqlite",
    "cockroachdb",
    "neo4j",
    "couchbase",
    "firebase",
    # streaming / big data
    "kafka",
    "spark",
    "flink",
    "hadoop",
    "airflow",
    "dbt",
    "fivetran",
    "dagster",
    "prefect",
    "kinesis",
    "pubsub",
    # ml / ai
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "sklearn",
    "xgboost",
    "huggingface",
    "langchain",
    "llama",
    "gpt",
    "openai",
    "anthropic",
    "claude",
    "llm",
    "llms",
    "transformers",
    "bert",
    "stable diffusion",
    "diffusers",
    "rag",
    "vector database",
    "pinecone",
    "weaviate",
    "chromadb",
    "milvus",
    "faiss",
    "openai api",
    "claude api",
    # data viz / bi
    "tableau",
    "looker",
    "power bi",
    "powerbi",
    "qlik",
    "sigma",
    "metabase",
    "superset",
    "mode",
    "google data studio",
    # design
    "figma",
    "sketch",
    "adobe xd",
    "photoshop",
    "illustrator",
    "after effects",
    "blender",
    "maya",
    "premiere pro",
    "indesign",
    "framer",
    "webflow",
    # devops monitoring
    "datadog",
    "splunk",
    "prometheus",
    "grafana",
    "new relic",
    "sentry",
    "pagerduty",
    "opsgenie",
    "elastic apm",
    # security
    "soc",
    "siem",
    "edr",
    "ids",
    "ips",
    "oauth",
    "oidc",
    "zero trust",
    "pentesting",
    "penetration testing",
    "burp suite",
    "metasploit",
    # erp / business systems
    "sap",
    "salesforce",
    "hubspot",
    "marketo",
    "netsuite",
    "oracle erp",
    "workday",
    "servicenow",
    "zendesk",
    # healthcare-specific
    "epic",
    "cerner",
    "meditech",
    "athena",
    "allscripts",
    # qa
    "selenium",
    "cypress",
    "playwright",
    "jest",
    "pytest",
    "mocha",
    "junit",
    "appium",
    "katalon",
    # game dev
    "unity",
    "unreal",
    "unreal engine",
    "godot",
    # ai infra / mlops
    "mlflow",
    "kubeflow",
    "sagemaker",
    "vertex ai",
    "azure ml",
    "wandb",
    "ray",
    "kserve",
    "triton",
]

ROLES = [
    "engineer",
    "developer",
    "architect",
    "scientist",
    "analyst",
    "consultant",
    "manager",
    "lead",
    "director",
    "specialist",
    "researcher",
    "intern",
    "designer",
    "coordinator",
    "administrator",
    "advisor",
    "associate",
    "executive",
]

SENIORITY = [
    "senior",
    "junior",
    "lead",
    "principal",
    "staff",
    "entry level",
    "new grad",
    "head of",
    "vice president",
    "vp",
]

# Top job-search "role bases" that pair well with seniority and skill
ROLE_BASES = [
    "software engineer",
    "software developer",
    "backend engineer",
    "frontend engineer",
    "full stack engineer",
    "data engineer",
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "ai engineer",
    "research scientist",
    "research engineer",
    "devops engineer",
    "sre",
    "site reliability engineer",
    "cloud engineer",
    "platform engineer",
    "security engineer",
    "mobile engineer",
    "ios engineer",
    "android engineer",
    "qa engineer",
    "test engineer",
    "automation engineer",
    "embedded engineer",
    "product manager",
    "product designer",
    "ux designer",
    "ui designer",
    "graphic designer",
    "marketing manager",
    "growth manager",
    "brand manager",
    "content strategist",
    "sales engineer",
    "account executive",
    "account manager",
    "customer success manager",
    "business analyst",
    "financial analyst",
    "operations analyst",
    "research analyst",
    "project manager",
    "program manager",
    "technical program manager",
    "support engineer",
    "solutions engineer",
    "solutions architect",
    "engineering manager",
    "design manager",
    "data science manager",
    "network engineer",
    "systems engineer",
    "systems administrator",
    "it support",
    "technical writer",
    "developer advocate",
    "registered nurse",
    "physician",
    "nurse practitioner",
    "physical therapist",
    "social worker",
    "teacher",
    "professor",
    "tutor",
    "recruiter",
    "talent acquisition",
    "hr generalist",
    "people operations",
    "controller",
    "auditor",
    "accountant",
    "tax accountant",
    "paralegal",
    "attorney",
    "legal counsel",
    "writer",
    "editor",
    "journalist",
    "warehouse worker",
    "delivery driver",
    "truck driver",
]

INDUSTRIES = [
    "tech",
    "fintech",
    "biotech",
    "edtech",
    "healthtech",
    "climate tech",
    "climate",
    "ai",
    "ml",
    "machine learning",
    "artificial intelligence",
    "cybersecurity",
    "security",
    "web3",
    "crypto",
    "blockchain",
    "gaming",
    "esports",
    "video games",
    "e-commerce",
    "ecommerce",
    "retail",
    "hospitality",
    "manufacturing",
    "logistics",
    "supply chain",
    "agriculture",
    "energy",
    "renewable energy",
    "oil and gas",
    "real estate",
    "legal",
    "accounting",
    "consulting",
    "education",
    "nonprofit",
    "government",
    "federal",
    "defense",
    "aerospace",
    "automotive",
    "media",
    "entertainment",
    "advertising",
    "marketing",
    "pharma",
    "pharmaceutical",
    "biotechnology",
    "medical devices",
    "saas",
    "b2b",
    "b2c",
    "deep tech",
    "data",
]

WORKSTYLE = [
    "remote",
    "wfh",
    "work from home",
    "fully remote",
    "remote first",
    "hybrid",
    "onsite",
    "in person",
    "in-person",
    "in office",
    "part time",
    "part-time",
    "full time",
    "full-time",
    "contract",
    "contractor",
    "freelance",
    "consulting",
    "temporary",
    "temp",
    "internship",
    "summer internship",
    "co-op",
    "coop",
    "entry level",
    "no experience",
    "no experience required",
    "no degree",
    "new grad",
    "recent graduate",
    "early career",
    "mid level",
    "mid-level",
    "weekend",
    "weekend only",
    "evening",
    "night shift",
    "day shift",
    "shift work",
    "shift",
]

COMPENSATION = [
    "high paying",
    "highest paying",
    "well paying",
    "best paying",
    "six figure",
    "6 figure",
    "100k",
    "150k",
    "200k",
    "250k",
    "300k",
    "100k+",
    "150k+",
    "200k+",
    "$100k",
    "$150k",
    "$200k",
    "$250k",
    "high salary",
    "competitive salary",
    "equity",
    "rsus",
    "with equity",
    "with bonus",
    "sign on bonus",
    "sign-on bonus",
]

EMPLOYER_TYPES = [
    "startup",
    "startups",
    "early stage startup",
    "seed stage",
    "series a",
    "series b",
    "series c",
    "growth stage",
    "late stage",
    "pre-ipo",
    "big tech",
    "faang",
    "magnificent seven",
    "mag 7",
    "fortune 500",
    "fortune 100",
    "small business",
    "small company",
    "mid sized company",
    "enterprise",
    "large enterprise",
    "federal government",
    "state government",
    "local government",
    "civil service",
    "nonprofit",
    "non-profit",
    "academia",
    "research lab",
    "national lab",
]

INTENT_PREFIXES = ["best", "top", "highest paid", "high paying", "easy"]

LOCATION_PATTERNS = ["jobs in {x}", "{x} jobs", "{x} careers"]
ROLE_PATTERNS_LOC = ["{role} in {x}", "{role} {x}", "{x} {role}"]

VISA = [
    "h1b sponsorship",
    "visa sponsorship",
    "h1b visa",
    "h-1b",
    "tn visa",
    "opt",
    "stem opt",
    "green card sponsorship",
    "us citizens only",
    "us citizen",
    "security clearance",
    "secret clearance",
    "top secret clearance",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_existing_keys():
    keys = set()
    for d in ["jobs_data", "jobs_data_linkedin", "jobs_data_jobstreet", "jobs_data_usajobs"]:
        for split in ["train", "eval"]:
            p = ROOT / d / f"{split}_queries_te3_1024.ids.json"
            if p.exists():
                with open(p) as f:
                    for q in json.load(f):
                        keys.add(q.strip().lower())
    for stem in ("aug_titles", "aug_combos"):
        p = ROOT / "unified_jobs" / f"{stem}_te3_1024.ids.json"
        if p.exists():
            with open(p) as f:
                for q in json.load(f):
                    keys.add(q.strip().lower())
    return keys


def load_locations_and_employers():
    """Return frequency-sorted lists of locations and employers from the corpus."""
    locs = Counter()
    emps = Counter()
    with open(ROOT / "unified_jobs" / "metadata.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ls = r.get("locations") or []
            if ls and ls[0]:
                loc = ls[0]
                parts = [p.strip() for p in loc.split(",")]
                if len(parts) >= 2:
                    locs[f"{parts[0]}, {parts[1]}".lower()] += 1
                else:
                    locs[loc.lower()] += 1
            emp_raw = r.get("source_slug") or ""
            if emp_raw and not any(c.isdigit() for c in emp_raw):
                emp = re.sub(r"[\-_]+", " ", emp_raw).strip()
                if emp and len(emp.split()) <= 4:
                    emps[emp] += 1
    return locs, emps


# -----------------------------------------------------------------------------
# Build the augmentation set
# -----------------------------------------------------------------------------


def main():
    existing = load_existing_keys()
    print(f"existing cache keys: {len(existing):,}", file=sys.stderr)

    locs, emps = load_locations_and_employers()
    top_locs = [loc for loc, _ in locs.most_common(500)]
    top_emps = [e for e, _ in emps.most_common(300)]

    aug = set()

    def add(q):
        q = norm(q)
        if q and q not in existing and len(q.split()) <= 8:
            aug.add(q)

    # 1) BARE SKILLS + "<skill> jobs"
    for s in SKILLS:
        add(s)
        add(f"{s} jobs")
        add(f"{s} developer")
        add(f"{s} engineer")
        add(f"{s} programmer")
        add(f"senior {s}")
        add(f"junior {s}")
        add(f"{s} consultant")
        add(f"{s} architect")
        add(f"{s} intern")
        add(f"{s} internship")

    # 2) SKILL x ROLE_BASES (combinations)
    high_value_skills = [
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "rust",
        "c++",
        "c#",
        "react",
        "angular",
        "vue",
        "node",
        "django",
        "flask",
        "fastapi",
        "spring",
        "aws",
        "gcp",
        "azure",
        "kubernetes",
        "docker",
        "terraform",
        "sql",
        "snowflake",
        "spark",
        "kafka",
        "airflow",
        "dbt",
        "tensorflow",
        "pytorch",
        "huggingface",
        "langchain",
        "llm",
        "tableau",
        "looker",
        "power bi",
        "figma",
        "ios",
        "android",
        "swift",
        "kotlin",
        "react native",
        "flutter",
        "salesforce",
        "sap",
        "selenium",
        "cypress",
    ]
    for s in high_value_skills:
        for r in ROLE_BASES[:50]:  # cap explosion
            add(f"{s} {r}")
            add(f"{r} {s}")

    # 3) SENIORITY x ROLE_BASES
    for sen in SENIORITY:
        for r in ROLE_BASES:
            add(f"{sen} {r}")

    # 4) SENIORITY x ROLE_BASES x WORKSTYLE (selective)
    for sen in ["senior", "junior", "lead", "principal", "staff"]:
        for r in ROLE_BASES[:40]:
            add(f"{sen} {r} remote")
            add(f"remote {sen} {r}")

    # 5) BARE ROLE BUCKETS + "<role> jobs"
    for r in ROLES + ROLE_BASES:
        add(r)
        add(f"{r} jobs")
        add(f"{r} careers")
        add(f"{r} openings")
        add(f"{r} roles")
        add(f"{r} opportunities")

    # 6) "jobs in <city>" / "<city> jobs" / "<role> in <city>"
    for loc in top_locs:
        add(f"jobs in {loc}")
        add(f"{loc} jobs")
        add(f"careers in {loc}")
        add(f"{loc} careers")
    # Top role x top loc
    for r in ROLE_BASES[:50]:
        for loc in top_locs[:80]:
            add(f"{r} in {loc}")
            add(f"{r} {loc}")

    # 7) "remote <role>" / "<role> remote"
    for r in ROLE_BASES:
        add(f"remote {r}")
        add(f"{r} remote")
        add(f"{r} work from home")
        add(f"{r} wfh")
    for r in ROLES:
        add(f"remote {r}")
        add(f"{r} remote")

    # 8) INDUSTRY + "<industry> jobs/careers"
    for ind in INDUSTRIES:
        add(ind)
        add(f"{ind} jobs")
        add(f"{ind} careers")
        add(f"jobs in {ind}")
        # industry x role
        for r in [
            "engineer",
            "developer",
            "scientist",
            "manager",
            "designer",
            "analyst",
            "researcher",
        ]:
            add(f"{ind} {r}")
            add(f"{r} in {ind}")

    # 9) WORKSTYLE modifiers (+role)
    for w in WORKSTYLE:
        add(w)
        add(f"{w} jobs")
        add(f"{w} careers")
        for r in ROLES[:8]:
            add(f"{w} {r}")
            add(f"{r} {w}")

    # 10) COMPENSATION (+ "<role> jobs")
    for c in COMPENSATION:
        add(c)
        add(f"{c} jobs")
        add(f"{c} remote jobs")
        for r in ["engineer", "developer", "manager"]:
            add(f"{c} {r} jobs")

    # 11) EMPLOYER_TYPES (+ role)
    for e in EMPLOYER_TYPES:
        add(e)
        add(f"{e} jobs")
        add(f"jobs at {e}")
        for r in ["engineer", "developer", "scientist", "manager", "designer", "analyst"]:
            add(f"{e} {r}")
            add(f"{r} at {e}")

    # 12) INTENT prefixes ("best <role>", "top <role>", etc.)
    for ip in INTENT_PREFIXES:
        for r in ROLE_BASES:
            add(f"{ip} {r}")
            add(f"{ip} {r} jobs")
        for r in ROLES:
            add(f"{ip} {r}")
            add(f"{ip} {r} jobs")

    # 13) VISA / clearance
    for v in VISA:
        add(v)
        add(f"{v} jobs")
        for r in ["engineer", "developer", "manager", "analyst"]:
            add(f"{v} {r}")

    # 14) "<role> near me" pattern
    for r in ROLE_BASES:
        add(f"{r} near me")
    for r in ROLES:
        add(f"{r} near me")

    # 15) Top-employer + role
    for emp in top_emps[:80]:
        for r in ["engineer", "developer", "manager", "designer", "analyst", "intern"]:
            add(f"{r} at {emp}")
            add(f"{emp} {r}")

    aug_list = sorted(aug)
    print(f"new aug queries: {len(aug_list):,}", file=sys.stderr)

    out_path = ROOT / "unified_jobs" / "head_torso_queries.jsonl"
    with open(out_path, "w") as f:
        for q in aug_list:
            f.write(json.dumps({"query": q}) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
