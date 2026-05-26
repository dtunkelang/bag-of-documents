#!/usr/bin/env python3
"""Tier 2 head/torso expansion: scale role x city, skill x role x city, and add
job-search functional categories.

Writes:
  unified_jobs/head_torso2_queries.jsonl
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    for stem in ("aug_titles", "aug_combos", "head_torso"):
        p = ROOT / "unified_jobs" / f"{stem}_te3_1024.ids.json"
        if p.exists():
            with open(p) as f:
                for q in json.load(f):
                    keys.add(q.strip().lower())
    return keys


def load_top_locs():
    locs = Counter()
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
    return locs


# 100 high-coverage role bases
TOP_ROLES = [
    "software engineer",
    "software developer",
    "backend engineer",
    "frontend engineer",
    "full stack engineer",
    "full-stack engineer",
    "data engineer",
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "ai engineer",
    "research scientist",
    "ml engineer",
    "devops engineer",
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
    "marketing manager",
    "growth manager",
    "content strategist",
    "content writer",
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
    "recruiter",
    "talent acquisition",
    "hr generalist",
    "controller",
    "auditor",
    "accountant",
    "paralegal",
    "attorney",
    "writer",
    "editor",
    "warehouse worker",
    "delivery driver",
    "truck driver",
    "executive assistant",
    "administrative assistant",
    "operations manager",
    "general manager",
    "store manager",
    "restaurant manager",
    "chef",
    "sous chef",
    "line cook",
    "bartender",
    "barista",
    "server",
    "electrician",
    "plumber",
    "carpenter",
    "mechanic",
    "graphic designer",
    "interior designer",
    "marketing specialist",
    "marketing analyst",
    "seo specialist",
    "digital marketing manager",
    "brand strategist",
    "social media manager",
    "data architect",
    "database administrator",
]

# 30 high-frequency skills that retrieve well
TOP_SKILLS = [
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "c++",
    "c#",
    "react",
    "angular",
    "vue",
    "node",
    "django",
    "aws",
    "azure",
    "gcp",
    "kubernetes",
    "docker",
    "terraform",
    "sql",
    "snowflake",
    "spark",
    "airflow",
    "dbt",
    "tensorflow",
    "pytorch",
    "huggingface",
    "langchain",
    "tableau",
    "looker",
    "figma",
    "salesforce",
    "sap",
    "ios",
    "android",
    "swift",
    "kotlin",
]

# Common job-search functional buckets (the "<category> jobs" pattern)
FUNCTIONS = [
    "engineering",
    "software",
    "data",
    "design",
    "product",
    "marketing",
    "sales",
    "finance",
    "accounting",
    "operations",
    "hr",
    "human resources",
    "legal",
    "support",
    "customer support",
    "customer service",
    "customer success",
    "research",
    "writing",
    "content",
    "editorial",
    "communications",
    "pr",
    "education",
    "teaching",
    "training",
    "consulting",
    "management",
    "administrative",
    "executive",
    "leadership",
    "strategy",
    "biology",
    "chemistry",
    "physics",
    "neuroscience",
    "nursing",
    "healthcare",
    "clinical",
    "medical",
    "construction",
    "manufacturing",
    "engineering trades",
    "skilled trades",
    "hospitality",
    "food service",
    "retail",
    "logistics",
    "supply chain",
    "transportation",
    "delivery",
    "creative",
    "media",
    "video",
    "audio",
    "photography",
    "fashion",
    "apparel",
]

INTENT_VERBS = ["hire", "hiring", "looking for", "find a", "find"]
PERIOD_MODIFIERS = ["2024", "2025", "2026", "this year", "now", "today", "asap"]


def main():
    existing = load_existing_keys()
    print(f"existing cache keys: {len(existing):,}")

    locs = load_top_locs()
    top_cities = [loc for loc, c in locs.most_common(200) if c >= 5]
    print(f"top cities considered: {len(top_cities)}")

    aug = set()

    def add(q):
        q = norm(q)
        if q and q not in existing and 1 <= len(q.split()) <= 8:
            aug.add(q)

    # 1) TOP_ROLES x TOP_CITIES (200 cities) — both orders + "in"
    for r in TOP_ROLES:
        for loc in top_cities:
            add(f"{r} {loc}")
            add(f"{r} in {loc}")
            add(f"{loc} {r}")
    print(f"  after role x city: {len(aug):,}")

    # 2) TOP_SKILLS x TOP_ROLES x TOP_CITIES (10 cities only — 30 x 50 x 10 = 15k)
    for s in TOP_SKILLS:
        for r in TOP_ROLES[:50]:
            for loc in top_cities[:10]:
                add(f"{s} {r} {loc}")
                add(f"{r} {s} {loc}")
    print(f"  after skill x role x city: {len(aug):,}")

    # 3) Functional category patterns
    for fn in FUNCTIONS:
        add(fn)
        add(f"{fn} jobs")
        add(f"{fn} careers")
        add(f"{fn} roles")
        add(f"jobs in {fn}")
        add(f"{fn} opportunities")
        for loc in top_cities[:30]:
            add(f"{fn} jobs in {loc}")
            add(f"{fn} jobs {loc}")
        add(f"remote {fn} jobs")
        add(f"{fn} remote")
    print(f"  after functional: {len(aug):,}")

    # 4) Intent verbs
    for v in INTENT_VERBS:
        for r in TOP_ROLES[:30]:
            add(f"{v} {r}")
        for fn in FUNCTIONS[:20]:
            add(f"{v} {fn} jobs")
    print(f"  after intent: {len(aug):,}")

    # 5) Year-modified job searches
    for mod in PERIOD_MODIFIERS:
        for r in TOP_ROLES[:25]:
            add(f"{r} {mod}")
            add(f"{r} jobs {mod}")
    print(f"  after year-mod: {len(aug):,}")

    # 6) Hybrid / onsite / remote x role x city
    for r in TOP_ROLES[:30]:
        for loc in top_cities[:30]:
            add(f"hybrid {r} {loc}")
            add(f"onsite {r} {loc}")
            add(f"remote {r} {loc}")
    print(f"  after workstyle x role x city: {len(aug):,}")

    # 7) "<role> hiring" / "<role> openings" + city
    for r in TOP_ROLES[:40]:
        for loc in top_cities[:20]:
            add(f"{r} hiring {loc}")
            add(f"{r} openings {loc}")
            add(f"{r} positions {loc}")
    print(f"  after hiring patterns: {len(aug):,}")

    aug_list = sorted(aug)
    print(f"\ntotal new aug: {len(aug_list):,}")

    out = ROOT / "unified_jobs" / "head_torso2_queries.jsonl"
    with open(out, "w") as f:
        for q in aug_list:
            f.write(json.dumps({"query": q}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
