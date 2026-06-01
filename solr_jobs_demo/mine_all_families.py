#!/usr/bin/env python3
"""Batch the role_family:other rescue miner across every target family.

Loads e5-small ONCE, then for each (family, query) runs the same pipeline as
mine_other_terms.py: dense KNN within role_family:other -> title n-grams ranked
by lift -> target-aware cross-family risk. Emits only the clean candidates
(OK = <=10% third-family spill, or low-risk ?? = <=20%) above a rescue-yield
floor, so the output is a paste-ready shortlist per family rather than raw dumps.

Usage:  python mine_all_families.py            # all families
        python mine_all_families.py sales hr_people_ops   # subset
"""

import sys

import mine_other_terms as m

# family_id -> natural-language query for the e5 dense lane. finance_accounting
# is omitted: already mined + shipped (commit 686899a).
FAMILIES: dict[str, str] = {
    "software_engineering": "software engineer developer programmer backend frontend full stack",
    "sales": "sales account executive business development quota",
    "marketing": "marketing brand content social media demand generation growth",
    "healthcare_clinical": "nurse physician clinical therapist medical doctor patient care",
    "skilled_trades_construction": "electrician plumber carpenter construction welder hvac technician",
    "customer_success_support": "customer success support help desk client services onboarding",
    "operations_admin": "operations administrative office manager executive assistant coordinator",
    "ai_ml": "machine learning artificial intelligence deep learning LLM model",
    "devops_sre_infra": "devops site reliability infrastructure platform cloud kubernetes",
    "product_management": "product manager product owner roadmap product strategy",
    "hr_people_ops": "human resources recruiter talent acquisition people operations",
    "project_program_management": "project manager program manager PMO scrum delivery",
    "security": "security cybersecurity information security SOC analyst threat",
    "creative_content": "content writer copywriter editor creative producer video",
    "design_ux": "designer UX UI product design graphic design researcher",
    "transportation_logistics": "logistics supply chain warehouse driver transportation fleet",
    "legal": "attorney lawyer legal counsel paralegal compliance contracts",
    "education_teaching": "teacher instructor professor tutor education faculty",
    "data_engineering": "data engineer ETL pipeline data platform warehouse",
    "data_science_ml": "data scientist machine learning statistics modeling experimentation",
    "consulting_strategy": "consultant strategy management consulting advisory transformation",
    "retail": "retail store associate sales floor merchandising cashier",
    "food_service_hospitality": "restaurant chef cook server hospitality hotel barista",
    "data_analytics": "data analyst business intelligence reporting dashboard BI insights",
    "manufacturing_production": "manufacturing production assembly machine operator plant quality",
    "healthcare_allied": "home health aide caregiver personal care direct support worker",
    "research_academic": "research scientist postdoc academic laboratory principal investigator",
    "nonprofit_social_services": "nonprofit social worker case manager community outreach program",
    "public_safety": "police firefighter security officer emergency dispatcher corrections",
    "healthcare_admin": "medical billing healthcare administration patient access medical records",
}

TOPN = 1000
BASELINE = 4000
LO, HI = 1, 3
MIN_SUPPORT = 6
SHOW = 60  # consider this many lifted grams per family
RISK_OK = 0.10  # <= this third-family spill = clean
RISK_MAYBE = 0.20  # <= this = worth a look
MIN_RESCUE = 8  # ignore grams that rescue fewer than this many 'other' docs


def mine_family(model, fam: str, query: str) -> list[dict]:
    qv = m._dense_qv(model, query)
    cand = m._knn_other(qv, TOPN)
    base = m._baseline_titles(BASELINE)
    cand_df = m._doc_freq(cand, LO, HI)
    base_df = m._doc_freq(base, LO, HI)
    nb, nc = max(1, len(base)), max(1, len(cand))

    rows = []
    for g, cf in cand_df.items():
        if cf < MIN_SUPPORT:
            continue
        lift = (cf / nc + 1e-4) / (base_df.get(g, 0) / nb + 1e-4)
        rows.append((lift, cf, g))
    rows.sort(reverse=True)

    keep = []
    for lift, _cf, g in rows[:SHOW]:
        dist = m._role_dist(g)
        total = sum(dist.values()) or 1
        other = dist.get("other", 0)
        if other < MIN_RESCUE:
            continue
        consistent = dist.get(fam, 0)
        risk = total - other - consistent
        risk_share = risk / total
        if risk_share > RISK_MAYBE:
            continue
        spill = (
            " ".join(
                f"{k}:{v}"
                for v, k in sorted(
                    ((v, k) for k, v in dist.items() if k not in (fam, "other")), reverse=True
                )[:3]
            )
            or "-"
        )
        keep.append(
            {
                "ngram": g,
                "lift": round(lift, 1),
                "rescue": other,
                "consistent": consistent,
                "risk": risk,
                "total": total,
                "risk_share": round(risk_share, 3),
                "spill": spill,
                "flag": "OK" if risk_share <= RISK_OK else "??",
            }
        )
    keep.sort(key=lambda r: -r["rescue"])
    return keep


def main() -> int:
    want = sys.argv[1:]
    fams = {k: v for k, v in FAMILIES.items() if not want or k in want}
    model = m._model()
    grand = 0
    for fam, query in fams.items():
        keep = mine_family(model, fam, query)
        resc = sum(r["rescue"] for r in keep)
        grand += resc
        print(
            f"\n{'=' * 78}\n## {fam}   (clean candidates rescue ~{resc} other docs)\n"
            f"   query: {query!r}"
        )
        if not keep:
            print("   (no clean candidates)")
            continue
        print(f"   {'flag':4} {'rescue':>6} {'risk%':>6} {'lift':>6}  ngram   [spill]")
        for r in keep:
            print(
                f"   {r['flag']:4} {r['rescue']:6d} {r['risk_share'] * 100:5.0f}% "
                f"{r['lift']:6.1f}  {r['ngram']!r}  [+{r['consistent']} {fam} | "
                f"{r['risk']}/{r['total']} -> {r['spill']}]"
            )
    print(
        f"\n{'=' * 78}\nGRAND TOTAL clean rescue potential: ~{grand} other docs "
        f"across {len(fams)} families (pre-dedup; grams overlap)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
