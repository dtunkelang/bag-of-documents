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
    # healthcare_allied — non-licensed direct-care workers (home health aides,
    # caregivers, personal-care aides, DSPs). MUST precede healthcare_clinical:
    # the clinical pattern used to grab these via "caregiver"/"home care"/"personal
    # care aide", which both mis-bucketed them (they're allied, not clinical) and,
    # because of plural/abbreviation gaps, dropped many to suppressed "other". The
    # negative lookahead keeps childcare nannies ("Child Caregiver") and corporate
    # roles *about* caregivers ("Caregiver Benefits Specialist") out of this bucket.
    (
        _ci(
            r"\b(home health aide|home health aid|HHA|CHHA|"
            r"personal care (aide|assistant|attendant)|"
            r"direct support (professional|worker|staff|personnel)|"
            r"home (health|care) (aide|aid|worker)|"
            r"(?<!child )caregivers?"
            r"(?![\s,]+(benefits|engagement|specialist|manager|operations|advocate|"
            r"recruiter|coordinator|relations|experience|support specialist|"
            r"success|enrollment|navigator|animal|pet|wellness)))\b"
        ),
        "healthcare_allied",
    ),
    # healthcare_clinical — high specificity to win against generic "engineer"
    (
        _ci(
            r"\b(registered nurse|RN|LPN|LVN|CNA|nurse practitioner|NP|"
            r"licensed practical nurse|licensed vocational nurse|"
            r"physician|MD\b|doctor|surgeon|pharmacist|dentist|veterinarian|"
            r"psychiatrist|psychologist|therapist|physical therapist|PT\b|"
            r"occupational therapist|OT\b|respiratory therapist|"
            r"nursing assistant|nurse aide|midwife|EMT|paramedic|"
            r"med(?:ication)? aide|sterile processing|"
            r"home care|caretaker|"
            r"(residential|substance abuse|mental health|behavioral health) counselor|"
            r"(psychiatric|mental health|behavioral health|clinical) "
            r"(clinician|specialist|therapist|supervisor)|"
            r"clinical (applications|systems|informatics|data|trials|nurse) "
            r"(specialist|coordinator|manager|analyst|associate)|"
            r"(veterinary|veterinarian) (assistant|technician|nurse)|"
            r"triage (nurse|specialist|supervisor|coordinator|associate)|"
            r"kinesiotherapist|kinesiologist|"
            r"patient (intake|care) (associate|specialist|coordinator|assistant)|"
            r"clinical (research )?coordinator|"
            r"prosthodontist|periodontist|orthodontist|endodontist|"
            r"radiologic technologist|sonographer|phlebotomist)\b"
        ),
        "healthcare_clinical",
    ),
    # healthcare_clinical — specialty physicians and clinical coordinators
    # the staffing-override audit found "Pediatric Neurologist", "General
    # Dermatologist", "Stroke and Sepsis Coordinator" etc. falling through to
    # "other". keep these patterns specific so they don't collide with
    # non-clinical "coordinator" / "specialist" roles.
    (
        _ci(
            r"\b(dermatologist|neurologist|cardiologist|oncologist|"
            r"pediatrician|endocrinologist|rheumatologist|gastroenterologist|"
            r"pulmonologist|hematologist|urologist|nephrologist|"
            r"anesthesiologist|radiologist|ophthalmologist|hospitalist|"
            r"speech[- ](language )?pathologist|SLP\b|audiologist|"
            r"dietitian|nutritionist|"
            r"(clinical|nursing|patient care|care|infection control|"
            r"trauma|wound( care)?|stroke|sepsis|cardiac|oncology|"
            r"perioperative|telemetry) coordinator)\b"
        ),
        "healthcare_clinical",
    ),
    (
        _ci(
            r"\b(medical (lab|laboratory) (technician|scientist)|"
            r"MLS\b|MLT\b|lab tech|"
            r"radiology tech|imaging tech|ultrasound tech|pharmacy tech|"
            r"dental hygienist|dental assistant|medical assistant|"
            r"patient transport|EKG tech|EEG tech|"
            r"(CT|MRI|x[- ]ray|cardiac|surgical|imaging|ultrasound|"
            r"cath lab|nuclear medicine|mammography|polysomnographic) "
            r"technologist)\b"
        ),
        "healthcare_allied",
    ),
    (
        _ci(
            r"\b(medical (biller|coder|coding|billing|scheduler|records)|"
            r"hospital administrat|patient services (coordinator|representative|rep)|patient access|"
            r"claims (handler|administrator|adjuster|representative|examiner|"
            r"analyst|specialist|processor|coordinator|manager)|"
            r"medical claims|"
            r"patient (navigator|advocate|liaison|coordinator|representative)|"
            r"practice (assistant|manager|coordinator|administrator))\b"
        ),
        "healthcare_admin",
    ),
    # security
    (
        _ci(
            r"\b(security engineer|security analyst|security architect|"
            r"infosec|cybersecurity|cyber security|penetration tester|pentester|"
            r"SOC analyst|threat (analyst|intel(?:ligence)?|hunt(?:ing)?)|incident response|"
            r"information security|"
            r"(security|cybersecurity|infosec) (network|director|manager|VP|head|lead)|"
            r"network security|"
            r"application security|appsec|GRC)\b"
        ),
        "security",
    ),
    # devops / sre / infra / platform
    (
        _ci(
            r"\b(DevOps|SRE\b|site reliability|platform engineer|infrastructure engineer|"
            r"cloud engineer|systems engineer|sysadmin|systems? administrator|"
            r"network engineer|kubernetes engineer|reliability engineer|"
            r"member of technical staff[ \-,]+\w*\s*(infrastructure|infra|platform|reliability))\b"
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
    # AI / ML — must come before data_science_ml and the broad
    # software_engineering catch-all so AI/ML titles land here rather than in
    # "other" (where they were suppressed from facets) or software_engineering.
    # Requires AI/ML to be a genuine technical or leadership *domain*, not an
    # incidental token: "(AI archetype)" game design, "AI Governance" legal
    # counsel, "AI Trainer" data-labeling gigs and "AI Video Artist" creative
    # spam are deliberately NOT matched (no bare-AI catch-all; nouns are limited
    # to technical/leadership functions).
    (
        _ci(
            # 1. unambiguous ML/AI domain signals anywhere in the title
            r"\b(machine learning|deep learning|reinforcement learning|"
            r"\bMLOps\b|ML ?ops|large language models?|\bLLMs?\b|"
            r"generative ai|gen ?ai|genai|agentic ai|computer vision|"
            r"neural network|\bNLP\b|"
            r"ML (engineer|scientist|researcher|research|architect|"
            r"infrastructure|platform|ops|lead)|"
            r"(ML/AI|AI/ML))\b"
            # 2. "<rank>[,] [of] [applied/generative] AI" leadership forms.
            #    "manager" is intentionally excluded so "Product Manager, AI",
            #    "Marketing Manager, AI", "Sales Manager AI" stay in their own
            #    functional families (only genuine AI leadership lands here).
            r"|\b(head|director|vp|svp|evp|chief|lead)[,\s]+"
            r"(of\s+)?(applied\s+|generative\s+|gen\s+)?a\.?i\.?\b"
            # 3. "AI <technical/leadership role-noun>" — noun stems (no trailing
            #    \b) so "AI Engineering", "AI Solutions", "AI Analytics" match.
            #    "product" is excluded so "AI Product Manager" stays in
            #    product_management.
            r"|\ba\.?i\.?\s+(engineer|scientist|research|architect|developer|"
            r"lead|special|consult|analy|director|strateg|"
            r"transformation|enablement|adoption|deployment|delivery|"
            r"implementation|solution|platform|infrastructure|"
            r"operation|ops|tech lead|team lead|intern|program)"
        ),
        "ai_ml",
    ),
    # data science — AI/ML engineering titles are claimed by ai_ml above; this
    # keeps data scientists and generic research/applied scientists. Analyst
    # titles are split out into data_analytics below — but this block comes
    # FIRST so "Data Scientist / Data Analyst" hybrids resolve to the scientist
    # family (the more specialized role) rather than to analytics.
    (
        _ci(r"\b(data scientist|research scientist|applied scientist)\b"),
        "data_science_ml",
    ),
    # analytics / BI — standalone data/business/BI/quant analysts, analytics
    # analysts and BI developers. Distinct from data_science_ml (model-building
    # scientists, claimed above) and data_engineering (pipeline/ETL/analytics
    # *engineers*, claimed earlier). Deliberately NARROW on two axes:
    #  - bare "analytics manager/lead/specialist" is NOT matched, so
    #    domain-specialized roles ("Marketing Analytics Manager", "Retail
    #    Analytics Manager", "People Analytics Specialist") stay in their
    #    functional family rather than being re-carved into analytics.
    #  - bare "reporting analyst" / "insights analyst" are NOT matched, since
    #    those pull financial/compliance reporting and market-research roles
    #    that belong in finance_accounting / marketing, not data/BI analytics.
    (
        _ci(
            r"\b(data analyst|business analyst|BI analyst|"
            r"business intelligence analyst|business intelligence developer|"
            r"analytics analyst|quantitative analyst|quant analyst)\b"
        ),
        "data_analytics",
    ),
    # civil / construction engineering — must come before generic
    # software_engineering, otherwise "engineer\b" catches "Highway Engineer",
    # "Construction Project Engineer", "Civil Engineering Manager", etc. and
    # routes them to software. routes to skilled_trades_construction.
    #
    # Generalised over four shapes (each title contains one of the
    # construction-discipline tokens listed in _DISC below):
    #   1. "<disc> [role-mod] engineer/engineering"      e.g. "Civil Project Engineer"
    #   2. "<disc> engineering <mgmt-rank>"               e.g. "Highway Engineering Manager"
    #   3. "engineer (...<disc>...)"                       e.g. "Project Engineer (Construction)"
    #   4. legacy: civil/structural EIT, construction inspector
    (
        _ci(
            # 1. discipline + optional role modifier + engineer|engineering
            r"\b(civil|structural|geotechnical|transportation|highway|"
            r"traffic|bridge|water resources|wastewater|environmental|"
            r"mining|petroleum|construction|roadway|coastal|hydraulic|"
            r"surveying|MEP|HVAC|drainage|pavement|earthworks|railway|"
            r"tunnel|dam|irrigation|mechanical|manufacturing|industrial)"
            r"( (project|design|field|site|technical|safety|quality|"
            r"principal|senior|junior|lead|staff|chief|assistant|"
            r"associate|supervising|consulting|review|inspection))?"
            r" (engineer|engineering)( intern)?\b|"
            # 2. <disc> engineering <mgmt-rank>
            r"\b(civil|structural|geotechnical|transportation|highway|"
            r"construction|MEP|HVAC|mechanical|manufacturing|industrial) engineering "
            r"(manager|director|lead|supervisor|head|chief|principal)\b|"
            # 3. engineer (...<disc>...) — discipline inside parens
            r"\bengineer\b\s*\([^)]*\b(construction|civil|structural|"
            r"highway|geotechnical|transportation|MEP|HVAC|"
            r"environmental|hydraulic)[^)]*\)|"
            # 4. legacy
            r"\b(civil EIT|structural EIT|construction inspector)\b"
        ),
        "skilled_trades_construction",
    ),
    # software engineering — broad, after specific eng specialties.
    # Also catches "Architect", "Engineering Manager/Director/VP" titles and
    # "Developer"-family standalone roles.
    (
        _ci(
            r"\b(software engineer|software developer|SDE\b|SWE\b|"
            r"backend engineer|back[- ]end|\bbackend\b|"
            r"frontend engineer|front[- ]end|\bfrontend\b|"
            r"full[- ]stack|\bfullstack\b|"
            r"web developer|mobile developer|iOS developer|"
            r"android developer|game developer|firmware engineer|embedded engineer|"
            r"QA engineer|test engineer|SDET|automation engineer|"
            r"engineering (manager|director|lead|head)|head of engineering|"
            r"director of engineering|VP of engineering|"
            r"software (architect|engineering)|solutions architect|"
            r"technical architect|principal architect|enterprise architect|"
            r"cloud architect|systems? architect|"
            r"SAP (technical|developer|engineer|architect|basis|HCM|ABAP|"
            r"IT (director|manager|consultant|lead))|"
            r"Salesforce (admin|administrator|developer|engineer|consultant|architect)|"
            r"Oracle (CPQ|EBS|cloud|consultant|developer|architect|engineer|admin)|"
            r"systems analyst|"
            r"IT (infrastructure|operations) (manager|director|head|lead|VP)|"
            r"(early career|new grad) (engineer|architecture|developer|opportunities)|"
            r"(android|iOS|mobile|web|game|firmware|embedded|backend|frontend) "
            r"(team\s+lead|tech\s+lead|lead)|"
            r"member of technical staff|"
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
            r"chief product officer|CPO\b|"
            r"product lead|growth lead|head of product)\b"
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
            r"copywriter|demand gen|email marketing|product marketing|PMM\b|"
            r"(earned|paid|owned) media|user acquisition|"
            r"analyst relations|influencer (marketing|relations|manager|director|strategist)|"
            r"head of (brand|marketing|growth|acquisition|content|communications)|"
            r"creative director,?\s+marketing|"
            r"social & influencer|social and influencer|"
            r"programmatic (lead|manager|director|specialist|trader|analyst|"
            r"planner|coordinator|associate)|"
            r"paid (search|media|social) (lead|manager|director|specialist|"
            r"analyst|coordinator|planner|associate)|"
            r"demand generation|"
            r"\bcommunications? (specialist|manager|director|lead|coordinator|"
            r"associate|consultant)|"
            r"digital accessibility|"
            r"field marketing|"
            r"(field|partner|channel) enablement|"
            r"(VP|director|manager|head),?\s+communications|"
            r"(VP|director|manager),?\s+(?:public )?(?:affairs|marketing|brand))\b"
        ),
        "marketing",
    ),
    # sales — includes pre-sales, market dev, partnerships, comma-titles
    (
        _ci(
            r"\b(account executive|AE\b|accounts executive|"
            r"sales (rep|representative|development|"
            r"manager|director|engineer|specialist|strategy|consultant|associate|"
            r"assistant|supervisor|operations|coordinator|lead)|"
            r"SDR\b|BDR\b|business development|biz dev|"
            r"inside sales|outside sales|sales associate|territory manager|"
            r"market development|"
            r"partnerships?\s+(manager|director|lead|principal|associate)|"
            r"senior director,?\s+partnerships|director,?\s+partnerships|"
            r"VP,?\s+partnerships|head of partnerships|"
            r"account (supervisor|coordinator)|"
            r"client (development|solutions|partner|relationship|engagement) "
            r"(associate|manager|director|lead|specialist)?|"
            r"client solutions director|"
            r"pre[- ]sales|presales|"
            r"VP,?\s+(.+\s+)?sales|director,?\s+(.+\s+)?sales|"
            r"manager,?\s+(.+\s+)?sales|head of sales|"
            r"head of (revenue|growth|business development|enterprise|customer)|"
            r"alliances? (manager|director|VP|lead|head|partner)|"
            r"(director|VP|head|manager),?\s+alliances|"
            r"(VIP|key account)\s+relationship\s+(manager|director|specialist|lead)|"
            r"account development (manager|director|VP|lead|associate)|"
            r"VP,?\s+account director|"
            r"sales executive (senior )?director|"
            r"district (partner|partnership) (specialist|manager|director)|"
            r"VP (of )?revenue|"
            # Portuguese/Spanish sales roles (the corpus has a large LatAm/BR
            # inventory the English-only patterns above were blind to).
            r"vendedor[ae]?s?|"
            r"(consultor|executiv[oa]|coordenador[ae]?)\s*(\(a\))?\s+de\s+vendas|"
            r"(ejecutiv[oa]|asesor[ae]?|representante|gerente)\s+de\s+ventas|"
            r"consultor\s*(\(a\))?\s+comercial|comercial externo|"
            r"agente comercial|asesor comercial|"
            r"de ventas|de vendas)\b"
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
            r"investment (banking|analyst|associate|banker|principal|director|manager)|"
            r"portfolio manager|underwriter|"
            r"equity research|deductions specialist|"
            r"corporate finance|head of (corporate )?finance|"
            r"banca product|bancassurance|"
            r"credit (analyst|manager|officer|risk)|"
            r"billing (clerk|specialist|coordinator|assistant)|"
            r"payroll (specialist|coordinator|clerk|assistant|"
            r"manager|consultant|administrator|analyst|director)|"
            r"branch banking|"
            r"risk (management )?(principal|director|VP|head)|"
            r"trade surveillance|"
            r"head of commercial compliance|"
            r"\bteller\b|fraud (insights|analyst|specialist|investigator|"
            r"operations|manager|director)|"
            r"SIU investigator|"
            r"internal controls|"
            r"(member|client) (banker|relationship specialist|relationship banker)|"
            r"private equity|hedge fund|"
            r"(finance|accounting|banking|payroll|treasury) (internship|intern)|"
            r"(commercial|loan|closing) (specialist|coordinator|officer)|"
            r"(bedbanks|originations?)\s+(analyst|director|manager|VP|head)|"
            r"director,?\s+origination|"
            r"budget (analyst|manager|director|specialist)|"
            r"trust (assurance|fund|operations) (analyst|manager|director|officer|specialist)|"
            r"collection(?:s)? (analyst|specialist|representative|coordinator|"
            r"manager|director|associate)|"
            r"balance sheet (manager|director|management|VP)|"
            r"pricing (analyst|manager|director|specialist|strategist|coordinator)|"
            r"retirement plan (specialist|manager|director|consultant|advisor)|"
            r"risk (?:and|&) compliance|"
            r"alternative distribution|"
            r"chief financial officer|CFO\b)\b"
        ),
        "finance_accounting",
    ),
    # legal
    (
        _ci(
            r"\b(attorney|lawyer|paralegal|legal counsel|compliance officer|"
            r"general counsel|associate attorney|law clerk|"
            r"(employment|tax|corporate|patent|trademark|trade|regulatory|"
            r"commercial|labor|privacy|litigation|securities|IP) counsel|"
            r"(appellate|litigation|patent|trademark|antitrust|securities|"
            r"corporate|tax|employment|immigration|family law|criminal|"
            r"intellectual property) (section|department|practice|group|"
            r"division|director|head|manager|partner|associate))\b"
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
            r"training (facilitator|coordinator|director|manager|lead|"
            r"specialist|consultant|developer)|"
            r"(technical|sales|customer|leadership) training|"
            r"performance management|"
            r"talent (business partner|business management|"
            r"development manager|operations|community|pool|"
            r"acquisition partner|management consultant)|"
            r"collective bargaining|labor relations|"
            r"(global )?head of (people|HR|talent|culture)|"
            r"chief people officer|CPO\b people|CHRO\b)\b"
        ),
        "hr_people_ops",
    ),
    # education
    (
        _ci(
            r"\b(teacher|tutor|instructor|professor|lecturer|"
            r"adjunct|teaching assistant|TA\b|education coordinator|"
            r"curriculum designer|curriculum (coordinator|developer|"
            r"director|specialist|manager)|"
            r"instructional designer|"
            r"principal of (?:education|school)|"
            r"(assistant|vice|school) principal|head of school|"
            r"head teacher|head of education)\b"
        ),
        "education_teaching",
    ),
    # research / academic
    (
        _ci(
            r"\b(postdoc|postdoctoral|research associate|research scientist|"
            r"PhD candidate|research fellow|lab manager|"
            r"(staff|principal|senior|chief|junior|quantum|materials|"
            r"computational|aerospace|biomedical|climate|earth|atmospheric|"
            r"physical|life|forensic|defense|nuclear|space|environmental) "
            r"(scientist|researcher)|"
            r"principal investigator|"
            r"ML research resident|research resident|"
            r"(machine learning|deep learning|NLP|computer vision) researcher)\b"
        ),
        "research_academic",
    ),
    # skilled trades + construction
    (
        _ci(
            r"\b(electrician|plumber|HVAC|carpenter|welder|machinist|"
            r"mechanic|technician|millwright|pipefitter|sheet metal|"
            r"construction worker|laborer|foreman|superintendent|"
            r"site supervisor|estimator|construction manager|"
            r"(QC|safety|civil|aviation|building|construction|site|"
            r"field|environmental|FDA|maintenance) inspector|"
            r"cladder|fabricator|"
            r"(?<!data\s)installer|(?<!cable\s)tradesperson|tradesman|"
            r"maintenance (planner|coordinator|technician|supervisor|"
            r"specialist|mechanic|manager|engineer)|"
            r"tool (and|&) die maker|"
            r"utility systems repairer(?:[ \-]+operator(?:\s+helper)?)?|"
            r"data center construction)\b"
        ),
        "skilled_trades_construction",
    ),
    # transportation / logistics
    (
        _ci(
            r"\b(driver|truck driver|CDL|delivery driver|courier|"
            r"dispatcher|warehouse|forklift|material handler|"
            r"logistics|supply chain|fleet manager|freight|"
            r"(airport )?ramp agent|airline operations|"
            r"baggage handler|ground crew|"
            r"shipping (and|&) receiving|shipping/receiving)\b"
        ),
        "transportation_logistics",
    ),
    # food / hospitality
    (
        _ci(
            r"\b(chef|sous chef|line cook|prep cook|baker|bartender|"
            r"server|waiter|waitress|host(?:ess)?|barista|"
            r"hotel|housekeep|concierge|front desk|hospitality|"
            r"restaurant manager|catering|food service worker|"
            r"food service (associate|manager|director|supervisor))\b"
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
            r"photographer|animator|art director|creative director|"
            r"(motion|apparel|fashion|textile|industrial|integrated|game|3D|"
            r"fabric|jewelry|footwear|costume) designer|"
            r"(audio|video|content|digital|creative|executive|associate|line|"
            r"senior|junior|staff|news|radio|podcast|broadcast|film|TV|"
            r"music|game) producer|"
            r"(character|concept|3D|2D|game|texture|VFX|"
            r"environment|technical|lighting|matte|storyboard) artist|"
            r"managing editor|"
            r"editorial (specialist|coordinator|director|manager|associate|QC))\b"
        ),
        "creative_content",
    ),
    # manufacturing / production
    (
        _ci(
            r"\b(machine operator|production (worker|associate|technician)|"
            r"assembly|assembler|manufacturing engineer|process engineer|"
            r"quality (engineer|inspector|technician|lead|supervisor|"
            r"coordinator|specialist|manager)|FSQA|food safety quality|"
            r"quality control (manager|director|inspector|technician|specialist)|"
            r"continuous improvement (coordinator|specialist|engineer|"
            r"manager|lead)|industrial engineer|"
            r"(CMM|CNC|PLC) (programmer|operator|technician|machinist)|"
            r"quality and food safety|food safety (and|&)? quality|"
            r"quality associate|"
            r"(section|line|shift|production) leader|"
            r"front load section|"
            r"food production worker|"
            r"operateur|opérateur\.?(?:trice)? de machine)\b"
        ),
        "manufacturing_production",
    ),
    # public safety
    (
        _ci(
            r"\b(police officer|deputy sheriff|firefighter|"
            r"corrections officer|security officer|security guard|"
            r"loss prevention officer|park ranger|safety officer|"
            r"border patrol|TSA|customs (and )?(border|immigration)|"
            r"immigration (officer|agent|inspector)|"
            r"federal (agent|investigator)|"
            r"firearm.* examiner|toolmark|forensic (examiner|specialist|analyst)|"
            r"aviation safety|FAA inspector)\b"
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
            r"strategic (planning|advisor|sourcing|partnerships|"
            r"initiatives|operations|alliance|advisor)|"
            r"(enterprise|solution|implementation|business) consultant|"
            r"corporate strategy (manager|director|associate|senior associate|"
            r"principal|lead)|"
            r"market intelligence|strategy & market|"
            r"chief of staff)\b"
        ),
        "consulting_strategy",
    ),
    # operations / admin — broad fallback before "other"
    (
        _ci(
            r"\b(operations manager|operations analyst|operations associate|"
            r"operations (lead|leader|director|coordinator|specialist|strategy)|"
            r"(VP|vice president) of operations|"
            r"senior associate,?\s+operations|"
            r"executive assistant|administrative assistant|admin assistant|"
            r"administrative (specialist|coordinator|associate)|"
            r"office (support|clerk|administrator)|"
            r"personal assistant|receptionist|secretary|"
            r"office manager|office coordinator|business operations|biz ops|"
            r"fleet (manager|administrator)|"
            r"procurement|sourcing manager|buyer\b|category manager|"
            r"facilities (manager|coordinator|specialist|engineer|project)|"
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


# SEEK / Jobstreet category prefix `[Category: <Top> / <Sub>]` appears verbatim
# at the start of ~15% of descriptions. It is structured taxonomy from the
# source ATS and is far more reliable than title regex for ambiguous engineer /
# coordinator / manager titles. Used in classify_role_family as an override
# when the title hits the software_engineering catch-all or matches nothing.

_CATEGORY_PREFIX = re.compile(r"^\s*\[Category:\s*([^\]]+)\]", re.IGNORECASE)

_CATEGORY_TOP_TO_ROLE: dict[str, str] = {
    "Accounting": "finance_accounting",
    "Administration & Office Support": "operations_admin",
    "Advertising, Arts & Media": "creative_content",
    "Banking & Financial Services": "finance_accounting",
    "Call Centre & Customer Service": "customer_success_support",
    "CEO & General Management": "c_level",
    "Community Services & Development": "nonprofit_social_services",
    "Construction": "skilled_trades_construction",
    "Consulting & Strategy": "consulting_strategy",
    "Design & Architecture": "design_ux",
    "Education & Training": "education_teaching",
    # SEEK Engineering covers civil/mechanical/electrical/etc., never software.
    # Default to trades; mechanical/electrical/etc. flip to manufacturing below.
    "Engineering": "skilled_trades_construction",
    "Farming, Animals & Conservation": "nonprofit_social_services",
    "Government & Defence": "public_safety",
    "Healthcare & Medical": "healthcare_clinical",
    "Hospitality & Tourism": "food_service_hospitality",
    "Human Resources & Recruitment": "hr_people_ops",
    "Information & Communication Technology": "software_engineering",
    "Insurance & Superannuation": "finance_accounting",
    "Legal": "legal",
    "Manufacturing, Transport & Logistics": "manufacturing_production",
    "Marketing & Communications": "marketing",
    "Mining, Resources & Energy": "manufacturing_production",
    "Real Estate & Property": "operations_admin",
    "Retail & Consumer Products": "retail",
    "Sales": "sales",
    "Science & Technology": "research_academic",
    "Sport & Recreation": "education_teaching",
    "Trades & Services": "skilled_trades_construction",
}

_CATEGORY_SUB_TO_ROLE: dict[str, str] = {
    # Engineering — non-civil disciplines route to manufacturing/production
    "Engineering / Mechanical Engineering": "manufacturing_production",
    "Engineering / Electrical/Electronic Engineering": "manufacturing_production",
    "Engineering / Process Engineering": "manufacturing_production",
    "Engineering / Industrial Engineering": "manufacturing_production",
    "Engineering / Chemical Engineering": "manufacturing_production",
    "Engineering / Aerospace Engineering": "manufacturing_production",
    "Engineering / Automotive Engineering": "manufacturing_production",
    "Engineering / Systems Engineering": "manufacturing_production",
    "Engineering / Materials Handling Engineering": "manufacturing_production",
    "Engineering / Maintenance": "manufacturing_production",
    "Engineering / Project Management": "project_program_management",
    "Engineering / Management": "project_program_management",
    # Construction — pm subcategories route to PM
    "Construction / Project Management": "project_program_management",
    "Construction / Contracts Management": "project_program_management",
    # ICT subcategories
    "Information & Communication Technology / Programme & Project Management": "project_program_management",
    "Information & Communication Technology / Management": "project_program_management",
    "Information & Communication Technology / Security": "security",
    "Information & Communication Technology / Networks & Systems Administration": "devops_sre_infra",
    "Information & Communication Technology / Help Desk & IT Support": "devops_sre_infra",
    "Information & Communication Technology / Database Development & Administration": "data_engineering",
    "Information & Communication Technology / Engineering - Network": "devops_sre_infra",
    "Information & Communication Technology / Engineering - Hardware": "devops_sre_infra",
    "Information & Communication Technology / Business/Systems Analysts": "data_analytics",
    "Information & Communication Technology / Consultants": "consulting_strategy",
    "Information & Communication Technology / Sales - Pre & Post": "sales",
    "Information & Communication Technology / Telecommunications": "devops_sre_infra",
    # Design & Architecture — building Architecture, not software
    "Design & Architecture / Architecture": "skilled_trades_construction",
    # Healthcare — pharma sales / admin
    "Healthcare & Medical / Sales": "sales",
    "Healthcare & Medical / Pharmaceuticals & Medical Devices": "sales",
    "Healthcare & Medical / Management": "healthcare_admin",
    # MTL — pull logistics-flavored subcats out of manufacturing
    "Manufacturing, Transport & Logistics / Warehousing, Storage & Distribution": "transportation_logistics",
    "Manufacturing, Transport & Logistics / Couriers, Drivers & Postal Services": "transportation_logistics",
    "Manufacturing, Transport & Logistics / Freight/Cargo Forwarding": "transportation_logistics",
    "Manufacturing, Transport & Logistics / Import/Export & Customs": "transportation_logistics",
    "Manufacturing, Transport & Logistics / Public Transport & Taxi Services": "transportation_logistics",
    "Manufacturing, Transport & Logistics / Rail & Maritime Transport": "transportation_logistics",
    "Manufacturing, Transport & Logistics / Road Transport": "transportation_logistics",
    # Real Estate — sales subcats
    "Real Estate & Property / Commercial Sales, Leasing & Property Mgmt": "sales",
    "Real Estate & Property / Residential Sales": "sales",
    "Real Estate & Property / Retail & Property Development": "sales",
    # Advertising/Arts/Media — marketing-flavored subcats
    "Advertising, Arts & Media / Media Strategy, Planning & Buying": "marketing",
    "Advertising, Arts & Media / Agency Account Management": "marketing",
}


def _role_from_description_category(desc: str) -> str | None:
    if not desc:
        return None
    m = _CATEGORY_PREFIX.match(desc)
    if not m:
        return None
    full = m.group(1).strip()
    if full in _CATEGORY_SUB_TO_ROLE:
        return _CATEGORY_SUB_TO_ROLE[full]
    top = full.split(" / ", 1)[0].strip()
    return _CATEGORY_TOP_TO_ROLE.get(top)


# Collapse rank/scope modifiers like "general/district/regional/assistant/..."
# when they sit between an industry-anchor and manager/supervisor/director/etc.
# Lets a single pattern (e.g. "restaurant manager") catch the wide family
# "restaurant <modifier> manager" instead of enumerating each variant.
_ROLE_MODIFIERS_RX = re.compile(
    r"\b((general|district|area|regional|national|global|"
    r"senior|junior|lead|principal|associate|assistant|deputy|chief|"
    r"staff|head|group|"
    r"vice president|VP|executive)\s+)+"
    r"(?=(manager|supervisor|director|coordinator|lead)\b)",
    re.IGNORECASE,
)


def _strip_role_modifiers(title: str) -> str:
    return _ROLE_MODIFIERS_RX.sub("", title)


def classify_role_family(title: str, desc: str = "") -> str:
    title = _strip_role_modifiers(title)
    for pat, family in ROLE_PATTERNS:
        if pat.search(title):
            # software_engineering is the broad engineer\b/developer\b catch-all.
            # Defer to the source-ATS category when the title is ambiguous.
            if family == "software_engineering":
                cat_family = _role_from_description_category(desc)
                if cat_family and cat_family != "software_engineering":
                    return cat_family
            return family
    cat_family = _role_from_description_category(desc)
    if cat_family:
        return cat_family
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
        "role_family": classify_role_family(title, desc),
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
