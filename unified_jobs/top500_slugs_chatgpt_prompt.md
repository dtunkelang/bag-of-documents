# ChatGPT prompt: label top 500 employer slugs with industry

Upload `top500_slugs_for_labeling.csv` to a ChatGPT conversation (recommend GPT-5 or GPT-4.1 with Code Interpreter / Advanced Data Analysis enabled), then paste the prompt below.

---

I have uploaded a CSV with 500 employer slugs from a job-postings corpus. Each row has:
- `rank` (popularity rank, 1 = most jobs)
- `slug` (employer slug)
- `source` (which job board the slug came from)
- `n_jobs` (number of postings for this employer)
- `top_titles` (top 5 distinct job titles, pipe-separated)
- `sample_description` (first 400 chars of a representative posting)

For each row, assign exactly one `industry` label from the fixed taxonomy below, plus a `confidence` of `high`, `medium`, or `low`.

**Rules**
- Use `slug` + `top_titles` as the primary signal. `sample_description` is a tiebreaker.
- Do not invent new labels. Use only the keys listed below.
- If genuinely unclear (multi-industry holding company, ambiguous slug, no signal), use `other` with `confidence=low`.
- A staffing/recruiting firm goes under `consulting_professional_services` regardless of which industries they staff for.
- A defense contractor goes under `defense_aerospace`, NOT `public_sector_government`.
- Restaurants/cafes/hotels = `hospitality_food_service`, NOT `consumer_brands` or `agriculture_food_production`.
- Food *manufacturers* (e.g. Kraft, Tyson) = `agriculture_food_production`; consumer *brands* (e.g. Coca-Cola, Unilever) = `consumer_brands`. Use judgment.

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
- **other** — An employer whose industry is unclear, multi-sector, or does not fit any of the listed categories.
