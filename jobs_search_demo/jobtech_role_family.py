#!/usr/bin/env python3
"""JobTech (Arbetsförmedlingen) occupation_field -> role_family crosswalk.

Every Swedish JobStream ad ships with the Arbetsförmedlingen occupation taxonomy
pre-attached; its broad bucket, `occupation_field`, is the Swedish analogue of
France Travail's ROME grand-domaine. We already capture that bucket's label in
the `department` field (download/fetch_jobtech_se.py), and it is a tiny,
authoritative controlled vocabulary (~21 fields), so we map it straight to a
role_family -- open-weight, deterministic, no LLM, no SSYK reconstruction needed.

The occupation_field label is what JobTech itself assigned, so within a field the
family is reliable. Genuinely homeless / cross-functional fields map to "other"
on purpose -- a wrong guess is worse than leaving the residual:
  - "Chefer och verksamhetsledare" (managers/executives): cross-functional, no home
  - "Yrken med teknisk inriktning" (technical occupations): spans every engineering
    discipline, no generic engineering family
  - "Sanering och renhållning" (sanitation/cleaning): homeless (cf. ROME K22/K23)
  - "Naturbruk" (agriculture/forestry): homeless (cf. ROME A*)
  - "Kropps- och skönhetsvård" (body/beauty care): personal-care, homeless
  - "Hantverk" (traditional crafts): too mixed (goldsmith..baker) to bucket safely
"""

from __future__ import annotations

# JobTech occupation_field label (exact, as stored in `department`) -> role_family.
SWED_FIELD: dict[str, str] = {
    "Hälso- och sjukvård": "healthcare_clinical",  # Health & medical care
    "Pedagogik": "education_teaching",  # Pedagogy / teaching
    "Yrken med social inriktning": "nonprofit_social_services",  # Social-orientation occupations
    "Administration, ekonomi, juridik": "operations_admin",  # Admin / economics / law (admin-dominant)
    "Försäljning, inköp, marknadsföring": "sales",  # Sales / purchasing / marketing (sales-dominant)
    "Transport, distribution, lager": "transportation_logistics",  # Transport / distribution / warehouse
    "Hotell, restaurang, storhushåll": "food_service_hospitality",  # Hotel / restaurant / catering
    "Bygg och anläggning": "skilled_trades_construction",  # Construction & civil engineering
    "Industriell tillverkning": "manufacturing_production",  # Industrial manufacturing
    "Data/IT": "software_engineering",  # Data / IT
    "Installation, drift, underhåll": "skilled_trades_construction",  # Install / operate / maintain
    "Säkerhet och bevakning": "security",  # Security & surveillance
    "Kultur, media, design": "creative_content",  # Culture / media / design
    "Naturvetenskap": "research_academic",  # Natural science
    "Militära yrken": "public_safety",  # Military occupations
    # --- intentionally NOT mapped (stay "other"): cross-functional / homeless ---
    "Chefer och verksamhetsledare": "other",  # Managers & executives
    "Yrken med teknisk inriktning": "other",  # Technical occupations (any engineering)
    "Sanering och renhållning": "other",  # Sanitation & cleaning
    "Naturbruk": "other",  # Agriculture / forestry / nature
    "Kropps- och skönhetsvård": "other",  # Body & beauty care
    "Hantverk": "other",  # Traditional crafts
}


def role_family_for_field(label: str | None) -> str:
    """Map a JobTech occupation_field label to a role_family ("other" if unknown)."""
    if not label:
        return "other"
    return SWED_FIELD.get(label.strip(), "other")
