#!/usr/bin/env python3
"""ROME (France Travail) -> role_family crosswalk (open-weight, deterministic).

Every France Travail offer is tagged by France Travail itself with a ROME 4.0
occupation code (1 letter grand-domaine + 4 digits, e.g. "N1110" = Magasinier).
This is the *authoritative* occupation label for the FR corpus -- far more
reliable than a lexical title->ESCO match -- so when an FR job carries a ROME
code we map it straight to a role_family, no LLM and no title heuristic.

The map is authored at the ROME *domaine professionnel* level (3 chars, e.g.
"M18"); ROME has 110 domaines, the granularity that aligns with role_family. A
1-letter grand-domaine fallback covers any unseen domaine. `role_family_for_rome`
resolves longest-prefix: 3-char domaine -> 1-char grand-domaine -> "other".

Precision note: France Travail assigned the code, so within a domaine the family
is reliable. Genuinely homeless domaines (agriculture A*, industrial/urban
cleaning K22/K23, funeral K26, sport L14, animal care) map to "other" on purpose
-- the corpus taxonomy has no home for them and a wrong guess is worse.
"""

from __future__ import annotations

# ROME domaine professionnel (3-char) -> role_family. Authored against the ROME
# 4.0 nomenclature (unix_domaine_professionnel_v460); labels in comments are the
# official domaine libelles.
ROME3: dict[str, str] = {
    # A - Agriculture, Peche, Espaces verts, Soins aux animaux (no home family)
    "A11": "other",  # Engins agricoles et forestiers
    "A12": "other",  # Espaces naturels et espaces verts
    "A13": "other",  # Etudes et assistance technique
    "A14": "other",  # Production
    "A15": "other",  # Soins aux animaux
    # B - Arts et Faconnage d'ouvrages d'art
    "B11": "creative_content",  # Arts plastiques
    "B12": "creative_content",  # Ceramique
    "B13": "design_ux",  # Decoration
    "B14": "creative_content",  # Fibres et papier
    "B15": "creative_content",  # Instruments de musique
    "B16": "creative_content",  # Metal, verre, bijouterie et horlogerie
    "B17": "other",  # Taxidermie
    "B18": "creative_content",  # Tissu et cuirs
    # C - Banque, Assurance, Immobilier
    "C11": "finance_accounting",  # Assurance
    "C12": "finance_accounting",  # Banque
    "C13": "finance_accounting",  # Finance
    "C14": "operations_admin",  # Gestion administrative banque et assurances
    "C15": "sales",  # Immobilier (agents immobiliers)
    # D - Commerce, Vente et Grande distribution
    "D11": "retail",  # Commerce alimentaire et metiers de bouche
    "D12": "retail",  # Commerce non alimentaire et de prestations de confort
    "D13": "retail",  # Direction de magasin de detail
    "D14": "sales",  # Force de vente
    "D15": "retail",  # Grande distribution
    # E - Communication, Media et Multimedia
    "E11": "creative_content",  # Edition et communication
    "E12": "creative_content",  # Images et sons
    "E13": "creative_content",  # Industries graphiques
    "E14": "marketing",  # Publicite
    # F - Construction, Batiment et Travaux publics
    "F11": "skilled_trades_construction",  # Conception et etudes
    "F12": "skilled_trades_construction",  # Conduite et encadrement de chantier
    "F13": "skilled_trades_construction",  # Engins de chantier
    "F14": "skilled_trades_construction",  # Extraction
    "F15": "skilled_trades_construction",  # Montage de structures
    "F16": "skilled_trades_construction",  # Second oeuvre
    "F17": "skilled_trades_construction",  # Travaux et gros oeuvre
    # G - Hotellerie-Restauration, Tourisme, Loisirs et Animation
    "G11": "food_service_hospitality",  # Accueil et promotion touristique
    "G12": "other",  # Animation d'activites de loisirs (homeless)
    "G13": "food_service_hospitality",  # Conception/commercialisation produits touristiques
    "G14": "food_service_hospitality",  # Gestion et direction
    "G15": "food_service_hospitality",  # Personnel d'etage en hotellerie
    "G16": "food_service_hospitality",  # Production culinaire
    "G17": "food_service_hospitality",  # Accueil en hotellerie
    "G18": "food_service_hospitality",  # Service
    # H - Industrie (all -> manufacturing_production)
    "H11": "manufacturing_production",  # Affaires et support technique client
    "H12": "manufacturing_production",  # Conception, recherche, etudes et developpement
    "H13": "manufacturing_production",  # Hygiene Securite Environnement industriels
    "H14": "manufacturing_production",  # Methodes et gestion industrielles
    "H15": "manufacturing_production",  # Qualite et analyses industrielles
    "H21": "manufacturing_production",  # Alimentaire
    "H22": "manufacturing_production",  # Bois
    "H23": "manufacturing_production",  # Chimie et pharmacie
    "H24": "manufacturing_production",  # Cuir et textile
    "H25": "manufacturing_production",  # Direction/encadrement fabrication et production
    "H26": "manufacturing_production",  # Electronique et electricite
    "H27": "manufacturing_production",  # Energie
    "H28": "manufacturing_production",  # Materiaux de construction, ceramique et verre
    "H29": "manufacturing_production",  # Mecanique, travail des metaux et outillage
    "H31": "manufacturing_production",  # Papier et carton
    "H32": "manufacturing_production",  # Plastique, caoutchouc
    "H33": "manufacturing_production",  # Preparation et conditionnement
    "H34": "manufacturing_production",  # Traitements thermiques et de surfaces
    # I - Installation et Maintenance
    "I11": "manufacturing_production",  # Encadrement
    "I12": "skilled_trades_construction",  # Entretien technique
    "I13": "manufacturing_production",  # Equipements de production, equipements collectifs
    "I14": "skilled_trades_construction",  # Equipements domestiques et informatique
    "I15": "skilled_trades_construction",  # Travaux d'acces difficile
    "I16": "skilled_trades_construction",  # Vehicules, engins, aeronefs
    # J - Sante
    "J11": "healthcare_clinical",  # Praticiens medicaux
    "J12": "healthcare_clinical",  # Praticiens medico-techniques
    "J13": "healthcare_allied",  # Professionnels medico-techniques
    "J14": "healthcare_allied",  # Reeducation et appareillage
    "J15": "healthcare_clinical",  # Soins paramedicaux (infirmiers, aides-soignants)
    # K - Services a la personne et a la collectivite (broad; per-domaine)
    "K11": "nonprofit_social_services",  # Accompagnement de la personne
    "K12": "nonprofit_social_services",  # Action sociale, socio-educative et socio-culturelle
    "K13": "nonprofit_social_services",  # Aide a la vie quotidienne
    "K14": "operations_admin",  # Conception/mise en oeuvre des politiques publiques
    "K15": "operations_admin",  # Controle public
    "K16": "other",  # Culture et gestion documentaire
    "K17": "public_safety",  # Defense, securite publique et secours
    "K18": "operations_admin",  # Developpement territorial et emploi
    "K19": "legal",  # Droit
    "K21": "education_teaching",  # Formation initiale et continue
    "K22": "other",  # Nettoyage et proprete industriels (homeless)
    "K23": "other",  # Proprete et environnement urbain (homeless)
    "K24": "research_academic",  # Recherche
    "K25": "public_safety",  # Securite privee
    "K26": "other",  # Services funeraires (homeless)
    # L - Spectacle
    "L11": "creative_content",  # Animation de spectacles
    "L12": "creative_content",  # Artistes - interpretes du spectacle
    "L13": "creative_content",  # Conception et production de spectacles
    "L14": "other",  # Sport professionnel (homeless)
    "L15": "creative_content",  # Techniciens du spectacle
    # M - Support a l'entreprise (split by domaine)
    "M11": "operations_admin",  # Achats
    "M12": "finance_accounting",  # Comptabilite et gestion
    "M13": "operations_admin",  # Direction d'entreprise
    "M14": "consulting_strategy",  # Organisation et etudes
    "M15": "hr_people_ops",  # Ressources humaines
    "M16": "operations_admin",  # Secretariat et assistance
    "M17": "marketing",  # Strategie commerciale, marketing et supervision des ventes
    "M18": "software_engineering",  # Systemes d'information et de telecommunication
    # N - Transport et Logistique (all -> transportation_logistics)
    "N11": "transportation_logistics",  # Magasinage, manutention, demenagement
    "N12": "transportation_logistics",  # Organisation de la circulation des marchandises
    "N13": "transportation_logistics",  # Personnel d'encadrement de la logistique
    "N21": "transportation_logistics",  # Personnel navigant transport aerien
    "N22": "transportation_logistics",  # Personnel sedentaire transport aerien
    "N31": "transportation_logistics",  # Personnel navigant transport maritime/fluvial
    "N32": "transportation_logistics",  # Personnel sedentaire transport maritime/fluvial
    "N41": "transportation_logistics",  # Personnel de conduite transport routier
    "N42": "transportation_logistics",  # Personnel d'encadrement transport routier
    "N43": "transportation_logistics",  # Personnel navigant transport terrestre
    "N44": "transportation_logistics",  # Personnel sedentaire transport ferroviaire
}

# 1-letter grand-domaine fallback for any unseen 3-char domaine.
ROME1: dict[str, str] = {
    "A": "other",
    "B": "creative_content",
    "C": "finance_accounting",
    "D": "retail",
    "E": "creative_content",
    "F": "skilled_trades_construction",
    "G": "food_service_hospitality",
    "H": "manufacturing_production",
    "I": "skilled_trades_construction",
    "J": "healthcare_clinical",
    "K": "nonprofit_social_services",
    "L": "creative_content",
    "M": "operations_admin",
    "N": "transportation_logistics",
}


def role_family_for_rome(rome: str | None) -> str:
    """Map a ROME code (e.g. 'N1110', 'M1607') to a role_family.

    Returns 'other' when the code is missing/malformed or maps to a domaine with
    no home family in the corpus taxonomy.
    """
    if not rome:
        return "other"
    rome = rome.strip().upper()
    if len(rome) < 3 or not rome[0].isalpha():
        return "other"
    if rome[:3] in ROME3:
        return ROME3[rome[:3]]
    if rome[:1] in ROME1:
        return ROME1[rome[:1]]
    return "other"
