#!/usr/bin/env python3
"""ISCO-08 -> role_family crosswalk (open-weight, deterministic).

ESCO occupations each carry an ISCO-08 code (e.g. "3155.1"); the corpus role
families (facets/taxonomy.ROLE_FAMILY) are a function-based axis. This module
maps the ISCO unit/minor group to the closest role family, so that a non-English
job whose title resolves to an ESCO occupation gets a role_family for free --
no LLM, no English-keyword heuristic.

The map is authored at the ISCO 3-digit minor-group level (the granularity that
aligns with role_family) with selective 4-digit overrides where a minor group
splits across families. `role_family_for_isco` resolves longest-prefix:
4-digit override -> 3-digit -> 2-digit fallback -> "other".

Precision note: the crosswalk is best-effort per group; precision of the overall
labeling is governed by the title->occupation match gate in classify_other_esco,
not by crosswalk perfection. Genuinely homeless ISCO groups (agriculture,
personal grooming, cleaning, sports) map to "other" on purpose -- the corpus
taxonomy has no home for them and a wrong guess is worse than leaving them.
"""

from __future__ import annotations

# 4-digit unit-group overrides: applied before the 3-digit default when a minor
# group genuinely splits across families.
ISCO4: dict[str, str] = {
    # 214 engineering professionals (no generic "engineering" family)
    "2142": "skilled_trades_construction",  # civil engineers
    "2141": "manufacturing_production",  # industrial & production engineers
    "2144": "manufacturing_production",  # mechanical engineers
    "2145": "manufacturing_production",  # chemical engineers
    "2146": "manufacturing_production",  # mining/metallurgical engineers
    # 216 architects/planners/designers
    "2163": "design_ux",  # product & garment designers
    "2166": "design_ux",  # graphic & multimedia designers
    # 242 administration professionals
    "2421": "consulting_strategy",  # management & organization analysts
    "2423": "hr_people_ops",  # personnel & careers professionals
    "2424": "hr_people_ops",  # training & staff development
    # 243 sales/marketing/PR professionals
    "2433": "sales",  # technical & medical sales (excl. ICT)
    "2434": "sales",  # ICT sales professionals
    # 134 professional services managers (split the coarse minor group)
    "1341": "nonprofit_social_services",  # child care services managers
    "1342": "healthcare_admin",  # health services managers
    "1343": "healthcare_allied",  # aged care services managers
    "1344": "nonprofit_social_services",  # social welfare managers
    "1345": "education_teaching",  # education managers (principals, rektor)
    "1346": "finance_accounting",  # financial & insurance branch managers
    # 252 database & network professionals
    "2521": "data_engineering",  # database designers & administrators
    "2522": "devops_sre_infra",  # systems administrators
    "2529": "security",  # database/network security
    # 263 social & religious professionals
    "2631": "finance_accounting",  # economists
    "2634": "healthcare_allied",  # psychologists
    # 333 business services agents
    "3333": "hr_people_ops",  # employment agents & contractors
    "3334": "sales",  # real estate agents
    # 335 regulatory government associates
    "3355": "public_safety",  # police inspectors & detectives
    # 341 legal/social/cultural associates
    "3411": "legal",  # legal & related associates
    # 343 artistic/cultural/culinary associates
    "3434": "food_service_hospitality",  # chefs
    "3432": "design_ux",  # interior designers & decorators
    # 351 ICT operations & user support
    "3512": "customer_success_support",  # ICT user support technicians
    # 352 telecom & broadcasting technicians
    "3521": "creative_content",  # broadcasting & audiovisual technicians
}

# 3-digit minor-group defaults.
ISCO3: dict[str, str] = {
    # 0 armed forces
    "011": "public_safety",
    "021": "public_safety",
    "031": "public_safety",
    # 1 managers
    "111": "operations_admin",
    "112": "operations_admin",
    "121": "operations_admin",
    "122": "sales",
    "131": "other",
    "132": "operations_admin",
    "133": "software_engineering",
    "134": "operations_admin",
    "141": "food_service_hospitality",
    "142": "retail",
    "143": "operations_admin",
    # 2 professionals
    "211": "research_academic",
    "212": "data_science_ml",
    "213": "research_academic",
    "214": "manufacturing_production",
    "215": "manufacturing_production",
    "216": "skilled_trades_construction",
    "221": "healthcare_clinical",
    "222": "healthcare_clinical",
    "223": "healthcare_clinical",
    "225": "healthcare_clinical",
    "226": "healthcare_clinical",
    "231": "education_teaching",
    "232": "education_teaching",
    "233": "education_teaching",
    "234": "education_teaching",
    "235": "education_teaching",
    "241": "finance_accounting",
    "242": "operations_admin",
    "243": "marketing",
    "251": "software_engineering",
    "252": "devops_sre_infra",
    "261": "legal",
    "262": "other",
    "263": "nonprofit_social_services",
    "264": "creative_content",
    "265": "creative_content",
    # 3 technicians & associate professionals
    "311": "manufacturing_production",
    "312": "skilled_trades_construction",
    "313": "manufacturing_production",
    "314": "research_academic",
    "315": "transportation_logistics",
    "321": "healthcare_allied",
    "322": "healthcare_allied",
    "323": "healthcare_allied",
    "324": "healthcare_allied",
    "325": "healthcare_allied",
    "331": "finance_accounting",
    "332": "sales",
    "333": "sales",
    "334": "operations_admin",
    "335": "operations_admin",
    "341": "nonprofit_social_services",
    "342": "other",
    "343": "creative_content",
    "351": "devops_sre_infra",
    "352": "devops_sre_infra",
    # 4 clerical support
    "411": "operations_admin",
    "412": "operations_admin",
    "413": "operations_admin",
    "421": "finance_accounting",
    "422": "customer_success_support",
    "431": "finance_accounting",
    "432": "transportation_logistics",
    "441": "operations_admin",
    # 5 service & sales
    "511": "food_service_hospitality",
    "512": "food_service_hospitality",
    "513": "food_service_hospitality",
    "514": "other",
    "515": "operations_admin",
    "516": "other",
    "521": "retail",
    "522": "retail",
    "523": "retail",
    "524": "sales",
    "531": "education_teaching",
    "532": "healthcare_allied",
    "541": "public_safety",
    # 6 skilled agricultural/forestry/fishery (no home family)
    "611": "other",
    "612": "other",
    "613": "other",
    "621": "other",
    "622": "other",
    # 7 craft & related trades
    "711": "skilled_trades_construction",
    "712": "skilled_trades_construction",
    "713": "skilled_trades_construction",
    "721": "skilled_trades_construction",
    "722": "manufacturing_production",
    "723": "skilled_trades_construction",
    "731": "manufacturing_production",
    "732": "manufacturing_production",
    "741": "skilled_trades_construction",
    "742": "skilled_trades_construction",
    "751": "manufacturing_production",
    "752": "manufacturing_production",
    "753": "manufacturing_production",
    "754": "manufacturing_production",
    # 8 plant & machine operators, assemblers
    "811": "manufacturing_production",
    "812": "manufacturing_production",
    "813": "manufacturing_production",
    "814": "manufacturing_production",
    "815": "manufacturing_production",
    "816": "manufacturing_production",
    "817": "manufacturing_production",
    "818": "manufacturing_production",
    "821": "manufacturing_production",
    "831": "transportation_logistics",
    "832": "transportation_logistics",
    "833": "transportation_logistics",
    "834": "skilled_trades_construction",
    "835": "transportation_logistics",
    # 9 elementary
    "911": "other",
    "912": "other",
    "921": "other",
    "931": "skilled_trades_construction",
    "932": "manufacturing_production",
    "933": "transportation_logistics",
    "941": "food_service_hospitality",
    "951": "other",
    "952": "retail",
    "961": "other",
    "962": "other",
}

# 2-digit sub-major fallback for any 3-digit group not in ISCO3.
ISCO2: dict[str, str] = {
    "01": "public_safety",
    "02": "public_safety",
    "03": "public_safety",
    "11": "operations_admin",
    "12": "operations_admin",
    "13": "operations_admin",
    "14": "operations_admin",
    "21": "research_academic",
    "22": "healthcare_clinical",
    "23": "education_teaching",
    "24": "operations_admin",
    "25": "software_engineering",
    "26": "creative_content",
    "31": "manufacturing_production",
    "32": "healthcare_allied",
    "33": "operations_admin",
    "34": "creative_content",
    "35": "devops_sre_infra",
    "41": "operations_admin",
    "42": "operations_admin",
    "43": "operations_admin",
    "44": "operations_admin",
    "51": "food_service_hospitality",
    "52": "retail",
    "53": "healthcare_allied",
    "54": "public_safety",
    "61": "other",
    "62": "other",
    "63": "other",
    "71": "skilled_trades_construction",
    "72": "manufacturing_production",
    "73": "manufacturing_production",
    "74": "skilled_trades_construction",
    "75": "manufacturing_production",
    "81": "manufacturing_production",
    "82": "manufacturing_production",
    "83": "transportation_logistics",
    "91": "other",
    "92": "other",
    "93": "manufacturing_production",
    "94": "food_service_hospitality",
    "95": "retail",
    "96": "other",
}


def role_family_for_isco(isco: str | None) -> str:
    """Map an ESCO ISCO-08 code (e.g. '3155.1', '7543.10.3') to a role_family.

    Returns 'other' when the code is missing or maps to a group with no home
    family in the corpus taxonomy.
    """
    if not isco:
        return "other"
    unit = isco.split(".")[0]  # drop ESCO decimal extensions
    if not unit.isdigit():
        return "other"
    if len(unit) >= 4 and unit[:4] in ISCO4:
        return ISCO4[unit[:4]]
    if len(unit) >= 3 and unit[:3] in ISCO3:
        return ISCO3[unit[:3]]
    if len(unit) >= 2 and unit[:2] in ISCO2:
        return ISCO2[unit[:2]]
    return "other"
