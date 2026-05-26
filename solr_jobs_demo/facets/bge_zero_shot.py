"""Zero-shot role_family + industry classification using bge-small embeddings.

We already have catalog embeddings for all 348k docs at
`unified_jobs/bge_catalog.vecs.fp16.npy`. For each role_family value, embed a
natural-language description and pick the label with highest cosine similarity.

Bge-small is symmetric for non-query use; both labels and docs are encoded with
no prefix.
"""

from __future__ import annotations

import numpy as np

# Label texts engineered to be coherent and distinctive in bge-small's space.
# Each describes the job role in a sentence that would semantically match the
# kinds of titles + descriptions we expect.
ROLE_LABEL_TEXTS: dict[str, str] = {
    "software_engineering": "Software engineer, software developer, backend engineer, frontend engineer, full-stack developer, mobile engineer, iOS or Android developer, web developer, software architect, or engineering manager building applications, websites, mobile apps, APIs, distributed systems, or platforms.",
    "data_engineering": "Data engineer, analytics engineer, ETL developer building data pipelines, data warehouses, lakehouses, batch and streaming jobs, dbt models, or analytics infrastructure for downstream analytics.",
    "data_science_ml": "Data scientist, machine learning engineer, AI engineer, ML researcher, applied scientist, or analytics professional building models, statistical analyses, A/B tests, dashboards, or recommendation systems.",
    "devops_sre_infra": "DevOps engineer, site reliability engineer SRE, platform engineer, infrastructure engineer, cloud engineer, systems administrator, or network engineer managing servers, Kubernetes, Terraform, CI/CD pipelines, and production reliability.",
    "security": "Cybersecurity engineer, information security analyst, application security AppSec engineer, penetration tester, SOC analyst, threat hunter, GRC compliance specialist, or security architect.",
    "design_ux": "Product designer, UX designer, UI designer, visual designer, interaction designer, or design manager creating user interfaces and experiences for digital products.",
    "product_management": "Product manager PM, product owner, group product manager GPM, or chief product officer CPO defining product roadmap, working with engineering and design.",
    "project_program_management": "Project manager, program manager, technical program manager TPM, scrum master, or delivery manager coordinating execution across teams on a defined timeline.",
    "marketing": "Marketing manager, growth marketer, brand manager, content marketer, SEO/SEM specialist, demand generation lead, email marketer, social media manager, product marketing manager PMM, or copywriter.",
    "sales": "Sales representative, account executive AE, business development representative BDR, sales development representative SDR, partnerships manager, territory manager, pre-sales engineer, or sales director closing deals and growing revenue.",
    "customer_success_support": "Customer success manager CSM, customer support representative, technical support engineer, account manager (post-sale), implementation engineer, solutions engineer or solutions consultant, or onboarding specialist supporting existing customers.",
    "operations_admin": "Operations manager, business operations analyst, executive assistant, administrative assistant, office manager, procurement specialist, facilities coordinator, program coordinator, or chief of staff handling internal company operations and administration.",
    "finance_accounting": "Accountant, controller, auditor, financial analyst, FP&A analyst, treasury analyst, tax associate, equity research analyst, investment banker, underwriter, bookkeeper, or CFO handling company finances and accounting.",
    "legal": "Attorney, lawyer, paralegal, legal counsel, general counsel, compliance officer, or contracts manager handling legal matters.",
    "hr_people_ops": "Recruiter, technical recruiter, talent acquisition specialist, human resources business partner HRBP, people operations manager, learning and development L&D specialist, compensation analyst, benefits administrator, or chief people officer.",
    "healthcare_clinical": "Registered nurse RN, LPN, nurse practitioner NP, physician, doctor MD, surgeon, dentist, pharmacist, veterinarian, psychologist, physical therapist PT, occupational therapist OT, respiratory therapist, paramedic, EMT, or midwife providing direct patient care.",
    "healthcare_allied": "Medical laboratory technician, radiology technologist, ultrasound technologist, pharmacy technician, dental hygienist, dental assistant, or medical assistant supporting clinical workflows.",
    "healthcare_admin": "Medical biller, medical coder, hospital administrator, patient access representative, scheduling coordinator, or healthcare operations specialist working in the administrative side of healthcare.",
    "education_teaching": "Teacher, professor, lecturer, instructor, tutor, teaching assistant, curriculum designer, principal, or education coordinator at K-12, higher ed, or other education settings.",
    "skilled_trades_construction": "Electrician, plumber, HVAC technician, carpenter, welder, machinist, mechanic, millwright, pipefitter, construction worker, foreman, superintendent, site supervisor, or estimator working in skilled trades or construction.",
    "transportation_logistics": "Truck driver CDL, delivery driver, courier, dispatcher, warehouse worker, forklift operator, material handler, fleet manager, logistics coordinator, or supply chain analyst.",
    "food_service_hospitality": "Chef, cook, baker, bartender, server, host, barista, hotel front desk, housekeeper, concierge, restaurant manager, or catering staff in food service or hospitality.",
    "retail": "Retail sales associate, cashier, store manager, stock associate, merchandiser, visual merchandiser, loss prevention specialist, or store associate working in a retail store.",
    "creative_content": "Writer, editor, journalist, content creator, copywriter, illustrator, graphic designer, video editor, videographer, photographer, animator, art director, or creative director producing creative content.",
    "research_academic": "Postdoctoral researcher, research associate, research scientist (in an academic or research lab context, not industry data science), lab manager, or PhD-level research fellow.",
    "manufacturing_production": "Machine operator, production worker, assembly line worker, manufacturing engineer, process engineer, quality control inspector, packing operator, or industrial engineer in a manufacturing or production setting.",
    "public_safety": "Police officer, deputy sheriff, firefighter, corrections officer, security officer or security guard, park ranger, or safety officer in a public safety or law enforcement role.",
    "nonprofit_social_services": "Social worker, case manager, community outreach coordinator, youth counselor, advocacy specialist, or nonprofit program coordinator working in social services or for a nonprofit organization.",
    "consulting_strategy": "Management consultant, strategy consultant, associate consultant, or strategic planning manager advising clients or internal business strategy.",
    "other": "A job that does not clearly fit any of the common job families above.",
}


def encode_labels(model) -> tuple[list[str], np.ndarray]:
    """Returns (label_keys, label_vecs) where label_vecs is (n_labels, dim)."""
    keys = list(ROLE_LABEL_TEXTS.keys())
    texts = [ROLE_LABEL_TEXTS[k] for k in keys]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return keys, vecs.astype(np.float32)


def classify_corpus(
    bge_catalog: np.ndarray, label_vecs: np.ndarray, keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, confidences) where labels are the picked role_family
    strings for each row, and confidences are the cosine similarities."""
    # bge_catalog: (N, dim) fp16 (already L2-normalized at encoding time).
    # Compute (N, n_labels) similarity in chunks to keep peak memory low.
    n = bge_catalog.shape[0]
    chunk = 50_000
    labels_idx = np.empty(n, dtype=np.int16)
    conf = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        block = bge_catalog[i : i + chunk].astype(np.float32)
        sims = block @ label_vecs.T  # (chunk, n_labels)
        labels_idx[i : i + chunk] = sims.argmax(axis=1)
        conf[i : i + chunk] = sims.max(axis=1)
    labels = np.array([keys[i] for i in labels_idx])
    return labels, conf


if __name__ == "__main__":
    import time

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    print("loading bge-small ...", flush=True)
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    keys, label_vecs = encode_labels(m)
    print(f"encoded {len(keys)} labels in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    bge = np.load(
        "/Users/dtunkelang/bagofdocs/unified_jobs/bge_catalog.vecs.fp16.npy", mmap_mode="r"
    )
    print(f"catalog: {bge.shape} {bge.dtype}", flush=True)
    labels, conf = classify_corpus(bge, label_vecs, keys)
    print(f"classified {len(labels):,} in {time.time() - t0:.1f}s", flush=True)
    from collections import Counter

    c = Counter(labels.tolist())
    total = sum(c.values())
    print("\nrole_family distribution (zero-shot):")
    for v, n in c.most_common():
        print(f"  {n:>6} ({100 * n / total:>5.1f}%)  {v}")
    print(f"\nmean confidence: {conf.mean():.3f}  median: {np.median(conf):.3f}")
    # Save
    out = "/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/role_family.zero_shot.npy"
    np.save(out, np.array([labels_idx for labels_idx in labels]))
    np.save("/Users/dtunkelang/bagofdocs/solr_jobs_demo/facets/role_family.conf.npy", conf)
    print(f"saved labels to {out}")
