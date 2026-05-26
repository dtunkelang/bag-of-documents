# Re-prompt: relabel 140 punted slugs

Upload `other140_for_relabel.csv` to a NEW ChatGPT chat (recommend GPT-5; not GPT-4.1, which was over-cautious last pass), then paste the prompt below.

---

I have uploaded a CSV with 140 employer slugs that a previous pass over-cautiously labeled "other". These are NOT placeholder rows — they are real, well-known companies that the prior model failed to commit on (it labeled OpenAI, Anthropic, Wells Fargo, Databricks etc. as "other"). Your job is to do better.

For each row, assign exactly one `industry` label from the taxonomy below, plus a `confidence` of `high` or `medium`. **Do not use "low". Do not use "other" unless the slug is genuinely a placeholder string with zero recognizable identity** (e.g. "lever-demo", "test", a random hash, a fully generic phrase). If you have any reasonable best-guess based on slug name + top_titles + sample_description, commit to it.

**Rules**
- Use slug name as a strong signal: if the slug clearly identifies a known company (e.g. `openai`, `wells-fargo`, `databricks`), label by what that company does.
- AI labs and developer-tools / cloud / data-platform companies → `tech_software_internet`.
- US/global banks (Wells Fargo, JPM, BofA, etc.) → `finance_banking`.
- Staffing/consulting/IT-services firms (TEKsystems, Apex Systems, etc.) → `consulting_professional_services`.
- Defense / mil-tech / autonomous-systems for military customers → `defense_aerospace`.
- Media events, conferences, content brands → `media_entertainment`.
- US federal/state agencies → `public_sector_government`.
- When the slug is unfamiliar but the titles are dominated by one industry's roles, label by the dominant industry.

**Output**
Return the full table as CSV with the original columns plus `industry` and `confidence` appended. No commentary, no markdown — just the CSV, ready to download.

---

## Industry taxonomy (28 labels)

- **tech_software_internet** — A software, internet, or SaaS company. Builds web platforms, mobile apps, developer tools, cloud services, AI products, consumer internet sites, or enterprise software. Includes B2B and B2C tech startups and large tech firms.
- **tech_hardware_semiconductors** — A hardware, electronics, semiconductor, or chip-design company. Manufactures or designs computer hardware, chips, networking gear, consumer electronics, or electronic components.
- **finance_banking** — A bank, investment bank, asset manager, broker-dealer, hedge fund, private equity firm, or wealth manager. Handles deposits, lending, capital markets, trading, or institutional finance.
- **finance_fintech** — A financial technology startup or company. Builds digital payments, neobanking apps, lending platforms, cryptocurrency exchanges, embedded finance, robo-advisory, or financial-data products.
- **finance_insurance** — An insurance company, reinsurer, or insurance broker. Sells life, health, property, casualty, auto, or commercial insurance, or underwrites and settles insurance claims.
- **healthcare_provider** — A hospital, clinic, medical practice, urgent care, telehealth service, dental office, mental health provider, or other direct patient-care organization.
- **healthcare_pharma_biotech** — A pharmaceutical, biotechnology, biopharma, drug-development, or life-sciences company. Researches, develops, manufactures, or commercializes drugs, vaccines, therapeutics, or biologic products.
- **healthcare_devices** — A medical-device company. Designs and manufactures diagnostic equipment, surgical instruments, implants, imaging machines, wearables for clinical use, or other healthcare hardware.
- **retail_ecommerce** — A retailer, e-commerce site, direct-to-consumer brand, marketplace, or specialty store. Sells physical or digital goods to consumers through stores or online.
- **consumer_brands** — A consumer packaged goods, fashion, beauty, food and beverage, or household products brand. Manufactures or markets branded consumer products distributed through retail channels.
- **media_entertainment** — A media, entertainment, publishing, news, broadcasting, music, film, sports, or streaming company. Produces or distributes content, news, video, audio, or live entertainment.
- **gaming** — A video game studio, mobile-game developer, or game publisher. Designs, develops, and publishes video games for console, PC, or mobile platforms.
- **automotive** — An automotive manufacturer, electric-vehicle company, auto-parts supplier, dealership, or mobility company. Makes, sells, or services cars, trucks, motorcycles, or vehicle technology.
- **energy_utilities** — An energy company: oil and gas, renewables, solar, wind, battery, electric utility, water utility, or grid operator. Generates, transmits, or delivers energy or utility services.
- **public_sector_government** — A federal, state, local, or international government agency, or a public-sector employer. Serves the public, regulates industries, or administers government programs. Excludes defense contractors.
- **defense_aerospace** — A defense contractor, aerospace manufacturer, military technology company, space company, satellite operator, or weapons-systems developer. Serves military, intelligence, or space customers.
- **nonprofit** — A nonprofit organization, charity, foundation, NGO, or advocacy group. Mission-driven; not primarily for-profit.
- **education_higher** — A university, college, graduate school, research institute, or other higher-education employer.
- **education_k12** — A K-12 school, school district, charter school, daycare, preschool, or tutoring service serving children and adolescents.
- **consulting_professional_services** — A management consulting firm, IT services firm, accounting firm, law-adjacent advisory, staffing agency, recruiting firm, or other professional-services employer that bills clients for expertise.
- **legal_services** — A law firm, legal practice, court, or legal-services provider. Practices law, represents clients, or provides legal advisory.
- **real_estate_construction** — A real estate developer, property manager, brokerage, REIT, construction company, general contractor, or homebuilder.
- **agriculture_food_production** — A farm, agricultural producer, agribusiness, food processor, or food manufacturer at the production level (not restaurants).
- **manufacturing** — A manufacturer of industrial goods, machinery, materials, chemicals, plastics, textiles, or general industrial products outside the automotive, aerospace, electronics, or pharma sectors.
- **telecommunications** — A telecommunications carrier, mobile network operator, broadband ISP, cable provider, or telecom infrastructure company.
- **transportation_logistics** — A trucking, shipping, logistics, freight, rail, airline, package delivery, supply-chain, or warehousing company. Moves goods or people for a living.
- **hospitality_food_service** — A hotel, restaurant, cafe, bar, catering company, hospitality brand, or travel-experience provider that serves guests directly.
- **other** — Reserved for placeholder slugs only (e.g. "lever-demo", "test"). Do NOT use for real companies with any identifiable industry.
