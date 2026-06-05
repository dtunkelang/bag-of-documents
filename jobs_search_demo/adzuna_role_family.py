#!/usr/bin/env python3
"""Adzuna category -> role_family crosswalk (open-weight, deterministic).

Every Adzuna posting carries a `category` from Adzuna's own ~28-category
taxonomy; we already store its localized label in the `department` field
(download/fetch_adzuna.py captures category.label). That label is the
source-assigned occupation bucket -- the cross-country analogue of ROME /
JobTech occupation_field -- so we map it straight to a role_family: no LLM, no
re-crawl, applies to docs already on disk.

Adzuna localizes the label per country (German "IT-Stellen", Italian
"Informatica/IT", ... all == the canonical "IT Jobs"), and occasionally serves a
label in the "wrong" language for a country. We therefore key the map on the
exact label STRING: a given string always denotes the same Adzuna category, so a
single flat dict resolves every language and auto-handles the cross-lane label
contamination. Unknown / generic / cross-functional labels fall through to
"other" on purpose (a wrong guess is worse than leaving the residual):
  - "Unknown" / "Other/General" / "Part time" (Teilzeit): no occupation signal
  - "Engineering" (Technikerstellen / Ingegneria / ingeniería): spans every
    engineering discipline, no generic engineering family
  - "Domestic help & Cleaning", "Energy/Oil & Gas", "Graduate": homeless
"""

from __future__ import annotations

# Exact Adzuna category label (any language, as stored in `department`) -> family.
ADZUNA_CAT: dict[str, str] = {
    # --- canonical English labels (served across several country lanes) ---
    "Accounting & Finance Jobs": "finance_accounting",
    "IT Jobs": "software_engineering",
    "Sales Jobs": "sales",
    "Customer Services Jobs": "customer_success_support",
    "Healthcare & Nursing Jobs": "healthcare_clinical",
    "Hospitality & Catering Jobs": "food_service_hospitality",
    "PR, Advertising & Marketing Jobs": "marketing",
    "Logistics & Warehouse Jobs": "transportation_logistics",
    "Trade & Construction Jobs": "skilled_trades_construction",
    "Admin Jobs": "operations_admin",
    "Teaching Jobs": "education_teaching",
    "Scientific & QA Jobs": "research_academic",
    "Manufacturing Jobs": "manufacturing_production",
    "Retail Jobs": "retail",
    "HR & Recruitment Jobs": "hr_people_ops",
    "Consultancy Jobs": "consulting_strategy",
    "Property Jobs": "sales",
    "Social work Jobs": "nonprofit_social_services",
    "Maintenance Jobs": "skilled_trades_construction",
    "Creative & Design Jobs": "creative_content",
    "Charity & Voluntary Jobs": "nonprofit_social_services",
    "Legal Jobs": "legal",
    "Domestic help & Cleaning Jobs": "other",
    # --- German ---
    "Stellen aus Gesundheitswesen & Pflege": "healthcare_clinical",
    "Stellen aus Buchhaltung & Finanzwesen": "finance_accounting",
    "IT-Stellen": "software_engineering",
    "Vertriebsstellen": "sales",
    "Stellen aus Logistik & Lagerhaltung": "transportation_logistics",
    "Verwaltungsstellen": "operations_admin",
    "Stellen aus Fertigung": "manufacturing_production",
    "Beraterstellen": "consulting_strategy",
    "Lehrberufe": "education_teaching",
    "Stellen aus Einzelhandel": "retail",
    "Stellen aus Personal & Personalbeschaffung": "hr_people_ops",
    "Stellen aus PR, Werbung & Marketing": "marketing",
    "Stellen aus Gastronomie & Catering": "food_service_hospitality",
    "Stellen aus Handel & Bau": "skilled_trades_construction",
    "Stellen aus Sozialarbeit": "nonprofit_social_services",
    "Kundendienststellen": "customer_success_support",
    "Immobilienstellen": "sales",
    "Stellen aus Wartung": "skilled_trades_construction",
    "Juristische Stellen": "legal",
    "Stellen aus Wissenschaft & Qualitätssicherung": "research_academic",
    "Stellen aus Kreation & Design": "creative_content",
    "Stellen aus Tourismus": "food_service_hospitality",
    "Gemeinnützige & ehrenamtliche Stellen": "nonprofit_social_services",
    # --- Dutch ---
    "Bouwkunde vacatures": "skilled_trades_construction",
    "Logistieke vacatures": "transportation_logistics",
    "Industriële vacatures": "manufacturing_production",
    "Gezondheidszorg Wetenschap en Verpleging vacatures": "healthcare_clinical",
    "Horeca en Catering vacatures": "food_service_hospitality",
    "Accounting en Financiële vacatures": "finance_accounting",
    "IT ICT vacatures": "software_engineering",
    "Detailhandel vacatures": "retail",
    "Sales vacatures": "sales",
    "Administratieve vacatures": "operations_admin",
    "PR Reclame en Marketing vacatures": "marketing",
    "Adviseur en Consultancy vacatures": "consulting_strategy",
    "Klantenservice vacatures": "customer_success_support",
    "Personeelszaken Human Resources vacatures": "hr_people_ops",
    "Onderhoud vacatures": "skilled_trades_construction",
    "Vacatures voor leraren": "education_teaching",
    "Maatschappelijk werk vacatures": "nonprofit_social_services",
    "QA en Kwaliteitsborging vacatures": "research_academic",
    "Juridische vacatures": "legal",
    "Creatieve Design vacatures": "creative_content",
    "Liefdadigheid en vrijwilligerswerk": "nonprofit_social_services",
    # --- Italian ---
    "Produzione/Industrie Manifatturiere": "manufacturing_production",
    "Logistica/Imballaggio E Magazzinaggio": "transportation_logistics",
    "Alberghi/Ristoranti/Bar": "food_service_hospitality",
    "Commerciale/Vendite": "sales",
    "Commercio Al Dettaglio/Retail": "retail",
    "Amministrazione/Segreteria": "operations_admin",
    "Sanità/Medicina": "healthcare_clinical",
    "Informatica/IT": "software_engineering",
    "Contabilità/Finanza": "finance_accounting",
    "Consulenza": "consulting_strategy",
    "Risorse Umane/HR": "hr_people_ops",
    "Customer Service/Call Center": "customer_success_support",
    "Pubblicità/Marketing/PR": "marketing",
    "Grafica/Creatività/Design": "creative_content",
    "Formazione/Istruzione": "education_teaching",
    "Elettrotecnica/Metalmeccanico": "manufacturing_production",
    "Affari Legali": "legal",
    "Scienza": "research_academic",
    "Immobiliare": "sales",
    "Turismo/Vacanze": "food_service_hospitality",
    "Edilizia": "skilled_trades_construction",
    # --- Spanish ---
    "Trabajos en logística y almacén": "transportation_logistics",
    "Trabajos en sanidad y salud": "healthcare_clinical",
    "Trabajos en ventas": "sales",
    "Trabajos en informática": "software_engineering",
    "Trabajos en fabricación y manufactura": "manufacturing_production",
    "Trabajos en hosteleria y restauración": "food_service_hospitality",
    "Trabajos en consultoría": "consulting_strategy",
    "Trabajos en administración": "operations_admin",
    "Trabajos en contabilidad y finanzas": "finance_accounting",
    "Trabajos en legal": "legal",
    "Trabajos en diseño y artes gráficas": "creative_content",
    "Trabajos de mantenimiento": "skilled_trades_construction",
    "Trabajos en construcción": "skilled_trades_construction",
    "Trabajos en tiendas": "retail",
    "Trabajos en marketing, publicidad y relaciones públicas": "marketing",
    "Trabajos en atención al cliente": "customer_success_support",
    "Trabajos en recursos humanos": "hr_people_ops",
    "Trabajos en educación": "education_teaching",
    "Trabajos en turismo": "food_service_hospitality",
    # --- French ---
    "Emplois Soins de santé et infirmiers": "healthcare_clinical",
    "Emplois Industrie et Construction": "skilled_trades_construction",
    "Emplois Fabrication": "manufacturing_production",
    "Emplois Vente": "sales",
    "Emplois Informatique": "software_engineering",
    "Emplois Comptabilité et Finance": "finance_accounting",
    "Emplois Distribution et Entrepôts": "transportation_logistics",
    "Emplois Hospitalité et Restauration": "food_service_hospitality",
    "Emplois Enseignement": "education_teaching",
    "Emplois Immobilier": "sales",
}


def role_family_for_adzuna_category(label: str | None) -> str:
    """Map an Adzuna category label to a role_family ("other" if unmapped)."""
    if not label:
        return "other"
    return ADZUNA_CAT.get(label.strip(), "other")
