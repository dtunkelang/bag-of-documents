"""Closed-vocabulary taxonomies for facet classification.

Each value is meant to be coherent (distinct meaning), distinctive (no near-
duplicate values), and close to exhaustive for the jobs corpora we have.
"""

ROLE_FAMILY: list[str] = [
    "software_engineering",
    "data_engineering",
    "data_science_ml",
    "data_analytics",
    "ai_ml",
    "devops_sre_infra",
    "security",
    "design_ux",
    "product_management",
    "project_program_management",
    "marketing",
    "sales",
    "customer_success_support",
    "operations_admin",
    "finance_accounting",
    "legal",
    "hr_people_ops",
    "healthcare_clinical",
    "healthcare_allied",
    "healthcare_admin",
    "education_teaching",
    "skilled_trades_construction",
    "transportation_logistics",
    "food_service_hospitality",
    "retail",
    "creative_content",
    "research_academic",
    "manufacturing_production",
    "public_safety",
    "nonprofit_social_services",
    "consulting_strategy",
    "other",
]

SENIORITY: list[str] = [
    "intern",
    "entry",
    "junior",
    "mid",
    "senior",
    "staff",
    "lead",
    "manager",
    "senior_manager",
    "director",
    "vp",
    "c_level",
    "not_specified",
]

REMOTE_MODE: list[str] = ["remote", "hybrid", "on_site", "not_specified"]

INDUSTRY: list[str] = [
    "tech_software_internet",
    "tech_hardware_semiconductors",
    "finance_banking",
    "finance_fintech",
    "finance_insurance",
    "healthcare_provider",
    "healthcare_pharma_biotech",
    "healthcare_devices",
    "retail_ecommerce",
    "consumer_brands",
    "media_entertainment",
    "gaming",
    "automotive",
    "energy_utilities",
    "public_sector_government",
    "defense_aerospace",
    "nonprofit",
    "education_higher",
    "education_k12",
    "consulting_professional_services",
    "legal_services",
    "real_estate_construction",
    "agriculture_food_production",
    "manufacturing",
    "telecommunications",
    "transportation_logistics",
    "hospitality_food_service",
    "other",
]

SALARY_BAND: list[str] = [
    "not_specified",
    "under_50k",
    "50k_75k",
    "75k_100k",
    "100k_150k",
    "150k_200k",
    "200k_300k",
    "300k_plus",
]

# Curated tech-stack vocabulary. Multi-value; empty for non-tech roles.
TECH_STACK: list[str] = [
    # Languages
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "C++",
    "C#",
    "Ruby",
    "PHP",
    "Swift",
    "Kotlin",
    "Scala",
    "R",
    "SQL",
    "Bash",
    # Frameworks / runtimes
    "React",
    "Vue",
    "Angular",
    "Next.js",
    "Node.js",
    "Django",
    "Flask",
    "FastAPI",
    "Spring",
    "Ruby on Rails",
    ".NET",
    "Laravel",
    # Data / streaming
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "Snowflake",
    "BigQuery",
    "Databricks",
    "Spark",
    "Kafka",
    "Airflow",
    "dbt",
    # Cloud
    "AWS",
    "GCP",
    "Azure",
    # DevOps / infra
    "Kubernetes",
    "Docker",
    "Terraform",
    "Ansible",
    "Jenkins",
    # ML
    "PyTorch",
    "TensorFlow",
    "Hugging Face",
    "LangChain",
    "OpenAI API",
    # Mobile / misc
    "iOS",
    "Android",
    "Linux",
    "Git",
]


def json_schema() -> dict:
    """OpenAI structured-output schema. All facets required; values constrained
    to the enums above. Returns minimal JSON suitable for atomic Solr update."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "role_family",
            "seniority",
            "remote_mode",
            "industry",
            "salary_band_usd_annual",
            "tech_stack",
        ],
        "properties": {
            "role_family": {"type": "string", "enum": ROLE_FAMILY},
            "seniority": {"type": "string", "enum": SENIORITY},
            "remote_mode": {"type": "string", "enum": REMOTE_MODE},
            "industry": {"type": "string", "enum": INDUSTRY},
            "salary_band_usd_annual": {"type": "string", "enum": SALARY_BAND},
            "tech_stack": {
                "type": "array",
                "items": {"type": "string", "enum": TECH_STACK},
            },
        },
    }


SYSTEM_PROMPT = """You are a meticulous classifier for a job-search index.

Given a job listing (title + a snippet of the description), output a strict JSON
object with these fields:

- role_family: the single best match from the enum. Pick the most specific.
- seniority: the seniority level from the enum. Use "not_specified" only when
  the listing genuinely doesn't indicate a level; otherwise infer from title
  and description content.
- remote_mode: pick "remote" only if the listing is fully remote;
  "hybrid" if it mixes remote with on-site time; "on_site" if it's clearly
  in-office or location-bound; "not_specified" when the listing doesn't say.
- industry: the employer's industry / sector (not the role's function).
  For staffing-agency listings, use the industry of the actual hiring company
  when discernible; otherwise "consulting_professional_services".
- salary_band_usd_annual: derive when salary is given.
  - Convert hourly to annual using 40 hrs/week * 52 weeks.
  - Convert non-USD currencies using rough parity (1 EUR = 1.1 USD, 1 GBP = 1.25 USD,
    1 INR = 0.012 USD, 1 SGD = 0.74 USD, 1 PHP = 0.018 USD, 1 AUD = 0.66 USD).
  - Use the midpoint of the stated range to pick a band.
  - Use "not_specified" when no salary is stated.
- tech_stack: list (possibly empty) of technologies explicitly named in the
  text, picking ONLY from the allowed vocabulary. Empty for non-tech roles.

Be strict about enum values. Do not invent new ones. Prefer "other" / "not_specified"
to a wrong-bucket guess."""
