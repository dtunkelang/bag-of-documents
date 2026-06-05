#!/usr/bin/env python3
"""Periodic refresh of the Solr jobs demo (Space dtunkelang/jobs-search).

Single orchestration script. Pulls the latest OpenApply + USAJobs postings,
rebuilds a te3-free unified catalog (RRF: BM25 + e5-small), reindexes Solr,
and (optionally) publishes the tarred core to the HF dataset + restarts the
Space. LinkedIn/JobStreet are intentionally DROPPED (frozen one-time scrapes);
see project_jobs_refresh_pipeline_plan.

Stages (run a contiguous range with --from-stage / --to-stage):
  0 pull    OpenApply (--openapply-source crawl [default]: run the crawler
            directly, ~25-30min same-day fresh; hf: download maintainer
            snapshot, fast but may lag ~1wk)
            + USAJobs API -> per-corpus titles/doc_ids/metadata via prep_open_apply
  1 unify   concatenate the 2 corpora -> OUT/{doc_ids,titles}.json,
            metadata.jsonl, source_index.json  (te3-free: NO vectors here)
  2 encode  e5-small-v2 over unified titles -> OUT/e5_small_catalog.vecs.fp16.npy
  3 facets  heuristics.classify_record -> facets/facets.jsonl
            (+ byproduct: facets/new_unlabeled_slugs.txt for later labeling)
  --- everything below mutates the live demo; gated behind --no-dry-run ---
  4 solr     start Solr, (re)create core, apply schema (+ industry field),
             push_docs, commit, verify. With --delta: incrementally upsert into
             the persistent core (add new + delete closed postings, no wipe) so
             the stage-6 Xet upload dedups (~MB/s vs full ~900MB). Solr id is a
             stable hash of the real doc id (not row position), so ids survive
             daily corpus reordering. Run a periodic full rebuild (no --delta) to
             reconcile in-place content edits.
  5 tar      COPYFILE_DISABLE=1 tar --no-xattrs the core
  6 upload   push solr_jobs_core.tar to dataset dtunkelang/jobs-demo
  7 deploy   upload space/app.py + restart Space + smoke check

Default is a DRY RUN: stages 0-3 only (data pipeline; no live mutation). Pass
--no-dry-run to allow stages 4-7.

Usage (this session's validation run):
  caffeinate -di .venv/bin/python jobs_search_demo/refresh.py \\
      --to-stage 3 --out-dir unified_jobs_v2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lang_detect import detect_lang  # sibling module; script dir is on sys.path

ROOT = Path("/Users/dtunkelang/bagofdocs")
PY = str(ROOT / ".venv" / "bin" / "python")

OPENAPPLY_REPO = "edwarddgao/open-apply-jobs"  # HF dataset, raw parquet (maintainer-uploaded)
OPENAPPLY_GIT = "https://github.com/edwarddgao/openapply"  # the crawler itself
OPENAPPLY_CLONE = ROOT / "openapply_repo"  # local checkout for --openapply-source crawl
EMBED_MODEL = "intfloat/e5-small-v2"
EMBED_OUT_NAME = "e5_small_catalog"  # -> {OUT}/e5_small_catalog.vecs.fp16.npy
EMBED_PREFIX = "passage: "  # e5 requires this on the document side
# Encode tuning: small chunks make each GPU->CPU transfer cheap, which both
# dodges the intermittent MPS waitUntilCompleted deadlock AND makes a kill+resume
# (via encode_st_catalog's per-chunk progress.json) cheap when one does happen.
EMBED_CHUNK_SIZE = 5000  # small per-chunk GPU->CPU transfer (~3.8MB) avoids the MPS deadlock
EMBED_BATCH_SIZE = 64  # e5-small is tiny; larger batch keeps the GPU busy (transfer size is per-chunk, not per-batch)
EMBED_STALL_SECS = 180  # no completed chunk within this -> assume MPS hang, kill + resume
EMBED_MAX_ATTEMPTS = 8

# (per-corpus dir, source_corpus tag) — order defines the unified index order.
# OpenApply first (the bulk + the refreshed source), USAJobs second, Adzuna third.
# Adzuna only contributes when ADZUNA_APP_ID/ADZUNA_APP_KEY are set AND its dir was
# populated this run (or carries over); unify skips any corpus dir with no artifacts.
CORPORA = [
    ("jobs_data", "jobs_data"),
    ("jobs_data_usajobs", "jobs_data_usajobs"),
    ("jobs_data_adzuna", "jobs_data_adzuna"),
    ("jobs_data_ats_extra", "jobs_data_ats_extra"),
    # Additional public sources (full-description; each only contributes when its
    # creds are set AND its dir was populated this run or carries over). The thin
    # snippet/title-only adapters (Jooble/Careerjet/Bundesagentur) are built but
    # deliberately NOT wired -- they'd dilute the embedding/snippet index.
    ("jobs_data_reed", "jobs_data_reed"),
    ("jobs_data_themuse", "jobs_data_themuse"),
    ("jobs_data_findwork", "jobs_data_findwork"),
    ("jobs_data_francetravail", "jobs_data_francetravail"),
    ("jobs_data_remoteok", "jobs_data_remoteok"),
    ("jobs_data_smartrecruiters", "jobs_data_smartrecruiters"),
    ("jobs_data_workable", "jobs_data_workable"),
    ("jobs_data_recruitee", "jobs_data_recruitee"),
    ("jobs_data_breezy", "jobs_data_breezy"),
    # JobTech (Arbetsförmedlingen, Sweden): keyless, full-description Swedish national
    # inventory with a pre-attached occupation taxonomy. Tagged lang=sv at unify so the
    # query-language gate scopes Swedish queries to it (and keeps it out of fr-scoped
    # results); English queries see it as the same mild low-rank noise as French.
    ("jobs_data_jobtech", "jobs_data_jobtech"),
    # Meta-aggregator LAST so unify can dedup it against everything above (see
    # DEDUP_AGAINST_PRIOR). Jooble re-lists jobs we already get from the ATS/Adzuna/
    # Reed/... sources, so only its genuinely-unique postings survive. Its docs are
    # snippet-grade (~300 chars) but that still feeds both the title+desc dense
    # vector and BM25, so the trade is "thinner body for more unique inventory".
    ("jobs_data_jooble", "jobs_data_jooble"),
]

# Corpora whose docs are DROPPED when they duplicate (by content signature) a
# posting already contributed by an earlier corpus. Listed last in CORPORA so they
# dedup against all the primary sources; primaries are never dropped (existing index
# composition is unchanged -- only the aggregator's re-lists are removed).
DEDUP_AGAINST_PRIOR = {"jobs_data_jooble"}

# User-maintained Greenhouse/Lever/Ashby tenant slugs that are NOT in OpenApply's
# cc_*_FINAL.txt lists (those ~15k tenants are already covered by stage-0's crawl).
# Drop a slug per line into cc_{ats}_EXTRA.txt to poll a company the harvest missed.
ATS_EXTRA_SLUGS = ROOT / "jobs_search_demo" / "extra_ats_slugs"

# Stage-4 Solr constants.
SOLR_URL = "http://localhost:8983"
SOLR_HOME = ROOT / "jobs_search_demo" / "solr_home"
JAVA_HOME = (
    "/opt/homebrew/opt/openjdk@21"  # run.sh's ${JAVA_HOME:-...} is buggy if JAVA_HOME is preset
)
CORE = os.environ.get("JOBS_CORE", "jobs")

# Artifact reuse: the slug->industry label CSVs are slug-keyed (corpus-independent),
# so they carry over from the previous build into the fresh OUT dir. INDUSTRY_CSV is the
# gated propagation; OVERRIDE_CSV is the hand-curated correction layer push_docs applies
# on top (see industry_filter.load_overrides). Both must reach STAGE for push_docs.
INDUSTRY_CSV = "slug_industry_labels_round2.csv"
OVERRIDE_CSV = "slug_industry_overrides.csv"
LEGACY_UNIFIED = ROOT / "unified_jobs"


def _seed_label_csvs(out: Path, tag: str) -> None:
    """Copy the slug->industry label CSVs from the persistent build into a fresh OUT dir
    so push_docs (which reads them from STAGE) sees both the gated labels and overrides."""
    for name in (INDUSTRY_CSV, OVERRIDE_CSV):
        src, dst = LEGACY_UNIFIED / name, out / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[{tag}] copied {name} into {out}", flush=True)


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    """Run a subprocess, streaming output; raise on non-zero exit."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    full_env = {**os.environ, **(env or {})}
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=full_env, check=True)


# ---------------------------------------------------------------------------
# Stage 0: pull
# ---------------------------------------------------------------------------
def _resolve_openapply_date(requested: str) -> str | None:
    """The OpenApply dataset is partitioned by daily snapshot date; each date is
    a full crawl of currently-open postings. 'latest' (default) picks the most
    recent date = the freshest complete catalog. 'all' returns None (pull every
    partition — heavy + highly redundant; dedup collapses it). A YYYY-MM-DD value
    pins a specific snapshot."""
    if requested == "all":
        return None
    if requested != "latest":
        return requested
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(OPENAPPLY_REPO, files_metadata=False)
    dates = sorted(
        {
            part.split("=")[1]
            for s in info.siblings
            if s.rfilename.endswith(".parquet")
            for part in s.rfilename.split("/")
            if part.startswith("date=")
        }
    )
    if not dates:
        sys.exit("[0] could not discover any date= partitions in OpenApply dataset")
    print(f"[0] discovered {len(dates)} snapshot dates; latest = {dates[-1]}", flush=True)
    return dates[-1]


def _pull_openapply_hf(oa_raw: Path, snapshot_date: str) -> None:
    """Fast path: download a daily snapshot from the maintainer's HF dataset.
    As fresh as their last upload (can lag ~1 week)."""
    from huggingface_hub import snapshot_download

    date = _resolve_openapply_date(snapshot_date)
    patterns = [f"**/date={date}/**/*.parquet"] if date else ["*.parquet", "**/*.parquet"]
    print(f"[0] downloading {OPENAPPLY_REPO} (date={date or 'ALL'}) ...", flush=True)
    snapshot_download(
        repo_id=OPENAPPLY_REPO,
        repo_type="dataset",
        local_dir=str(oa_raw),
        allow_patterns=patterns,
    )


def _crawl_openapply(oa_raw: Path, workers: int) -> None:
    """Fresh path: run the OpenApply crawler directly (~15 min). Hits public
    Greenhouse/Lever/Ashby APIs across the committed slugs/cc_*_FINAL.txt lists
    (no creds, no Common Crawl harvest). Emits the same parquet schema the HF
    dataset uses, so prep_open_apply consumes it unchanged. Same-day fresh."""
    if OPENAPPLY_CLONE.exists():
        run(["git", "-C", OPENAPPLY_CLONE, "pull", "--ff-only"])
    else:
        run(["git", "clone", "--depth", "1", OPENAPPLY_GIT, OPENAPPLY_CLONE])

    jsonl = OPENAPPLY_CLONE / "jobs.jsonl"
    # oa_adapter reads slugs/cc_*_FINAL.txt relative to its own dir -> cwd=clone.
    run(
        [PY, "oa_adapter.py", "--workers", str(workers), "--out", "jobs.jsonl"], cwd=OPENAPPLY_CLONE
    )

    # A fresh crawl is a single current snapshot, so clear stale parquet first.
    if oa_raw.exists():
        shutil.rmtree(oa_raw)
    oa_raw.mkdir(parents=True, exist_ok=True)
    run([PY, "scripts/jsonl_to_parquet.py", jsonl, oa_raw], cwd=OPENAPPLY_CLONE)


def _crawl_ats_extra(out_raw: Path) -> bool:
    """Poll Greenhouse/Lever/Ashby for the user-maintained extra tenant slugs,
    minus any already in OpenApply's FINAL lists (those are covered by the main
    stage-0 crawl, so subtracting them avoids both wasted API calls and
    cross-corpus duplicates). Reuses the cloned oa_adapter.py so the parsing is
    byte-identical to the main crawl. Returns True if a parquet was produced."""
    # The main crawl clones this; if stage-0 pulled OpenApply from HF instead,
    # clone now -- we need both oa_adapter.py and the FINAL lists to subtract against.
    if not OPENAPPLY_CLONE.exists():
        run(["git", "clone", "--depth", "1", OPENAPPLY_GIT, OPENAPPLY_CLONE])

    final_dir = OPENAPPLY_CLONE / "slugs"
    tmp = out_raw.parent / "_slugs"  # temp slug-dir for oa_adapter (cc_{ats}_EXTRA.txt)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    ats_with_slugs: list[str] = []
    total = 0
    for ats in ("greenhouse", "lever", "ashby"):
        src = ATS_EXTRA_SLUGS / f"cc_{ats}_EXTRA.txt"
        if not src.exists():
            continue
        want = [
            ln.strip()
            for ln in src.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        final_path = final_dir / f"cc_{ats}_FINAL.txt"
        covered = (
            {ln.strip().lower() for ln in final_path.read_text().splitlines() if ln.strip()}
            if final_path.exists()
            else set()
        )
        fresh = [s for s in want if s.lower() not in covered]
        dropped = len(want) - len(fresh)
        if dropped:
            print(f"[0] ats-extra {ats}: {dropped} slug(s) already in FINAL, skipped", flush=True)
        if fresh:
            (tmp / f"cc_{ats}_EXTRA.txt").write_text("\n".join(fresh) + "\n")
            ats_with_slugs.append(ats)
            total += len(fresh)

    if not ats_with_slugs:
        print("[0] ats-extra: no new tenants to poll (empty or all in FINAL); skipping", flush=True)
        return False

    print(f"[0] ats-extra: polling {total} tenant(s) across {ats_with_slugs}", flush=True)
    run(
        [
            PY,
            "oa_adapter.py",
            "--slug-dir",
            str(tmp),
            "--suffix",
            "EXTRA",
            "--ats",
            ",".join(ats_with_slugs),
            "--workers",
            "8",
            "--out",
            "jobs_extra.jsonl",
        ],
        cwd=OPENAPPLY_CLONE,
    )
    if out_raw.exists():
        shutil.rmtree(out_raw)
    out_raw.mkdir(parents=True, exist_ok=True)
    run(
        [PY, "scripts/jsonl_to_parquet.py", OPENAPPLY_CLONE / "jobs_extra.jsonl", out_raw],
        cwd=OPENAPPLY_CLONE,
    )
    return True


def _pull_api_source(args, *, corpus_dir: str, label: str, have_creds: bool, fetch_cmd: list):
    """Cred-gated fetch -> prep for a simple single-snapshot API source.

    Mirrors the Adzuna block: when creds are present and not --skip-download, do a
    fresh fetch (clearing stale raw first so closed postings don't linger), then
    prep. Otherwise reuse existing raw if any, else skip the corpus entirely. The
    same canonical 13-col parquet schema -> the same prep -> the same downstream
    stages, so unify just picks up the extra corpus dir.
    """
    d = ROOT / corpus_dir
    raw = d / "raw"
    have_raw = raw.exists() and any(raw.glob("*.parquet"))
    if args.skip_download or not have_creds:
        if not have_raw:
            print(
                f"[0] no {label} creds and no raw parquet to reuse; skipping {label} corpus",
                flush=True,
            )
            return
        why = "--skip-download" if args.skip_download else f"no {label} creds"
        print(f"[0] {why}: reusing existing {label} parquet under {raw}", flush=True)
    else:
        if raw.exists():
            shutil.rmtree(raw)
        run(fetch_cmd)
    run([PY, "download/prep_open_apply.py", "--raw-dir", raw, "--out-dir", d, "--sample-n", "0"])


def stage_pull(args) -> None:
    # --- OpenApply ---
    oa_raw = ROOT / "jobs_data" / "raw"
    if args.skip_download:
        print(f"[0] --skip-download: reusing existing parquet under {oa_raw}", flush=True)
    elif args.openapply_source == "crawl":
        _crawl_openapply(oa_raw, args.crawl_workers)
    else:
        _pull_openapply_hf(oa_raw, args.snapshot_date)
    run(
        [
            PY,
            "download/prep_open_apply.py",
            "--raw-dir",
            oa_raw,
            "--out-dir",
            ROOT / "jobs_data",
            "--sample-n",
            str(args.openapply_sample_n),  # 0 = keep all
        ]
    )

    # --- USAJobs ---
    usa_raw = ROOT / "jobs_data_usajobs" / "raw"
    have_creds = bool(os.environ.get("USAJOBS_EMAIL") and os.environ.get("USAJOBS_API_KEY"))
    have_raw = usa_raw.exists() and any(usa_raw.glob("*.parquet"))
    if args.skip_download or not have_creds:
        if not have_raw:
            sys.exit(
                "[0] no USAJobs raw parquet to reuse and no API creds; set USAJOBS_EMAIL + USAJOBS_API_KEY"
            )
        why = "--skip-download" if args.skip_download else "no USAJOBS creds"
        print(f"[0] {why}: reusing existing USAJobs parquet under {usa_raw}", flush=True)
    else:
        run([PY, "download/fetch_usajobs.py", "--out-dir", usa_raw, "--max-pages", "0"])
    # USAJobs raw parquet shares prep_open_apply's expected field names, so the
    # same prep produces titles/doc_ids/metadata for it.
    run(
        [
            PY,
            "download/prep_open_apply.py",
            "--raw-dir",
            usa_raw,
            "--out-dir",
            ROOT / "jobs_data_usajobs",
            "--sample-n",
            "0",  # keep all USAJobs
        ]
    )

    # --- Adzuna (optional: only when creds are set) ---
    # Aggregator inventory (job boards / recruiters / non-ATS employers) the
    # OpenApply ATS crawl and federal USAJobs don't reach. Recency-first fetch,
    # same canonical parquet schema -> same prep -> same downstream stages.
    adz_dir = ROOT / "jobs_data_adzuna"
    adz_raw = adz_dir / "raw"
    have_adz_creds = bool(os.environ.get("ADZUNA_APP_ID") and os.environ.get("ADZUNA_APP_KEY"))
    have_adz_raw = adz_raw.exists() and any(adz_raw.glob("*.parquet"))
    if args.skip_download or not have_adz_creds:
        if not have_adz_raw:
            print(
                "[0] no Adzuna creds and no raw parquet to reuse; "
                "skipping Adzuna corpus (set ADZUNA_APP_ID + ADZUNA_APP_KEY to enable)",
                flush=True,
            )
        else:
            why = "--skip-download" if args.skip_download else "no Adzuna creds"
            print(f"[0] {why}: reusing existing Adzuna parquet under {adz_raw}", flush=True)
            run(
                [
                    PY,
                    "download/prep_open_apply.py",
                    "--raw-dir",
                    adz_raw,
                    "--out-dir",
                    adz_dir,
                    "--sample-n",
                    "0",
                ]
            )
    else:
        # Fresh fetch is one current snapshot; clear stale parquet so deleted
        # postings don't linger (mirrors the crawl path's rmtree).
        if adz_raw.exists():
            shutil.rmtree(adz_raw)
        run(
            [
                PY,
                "download/fetch_adzuna.py",
                "--out-dir",
                adz_raw,
                "--countries",
                args.adzuna_countries,
                "--max-pages",
                str(args.adzuna_max_pages),
                "--max-days-old",
                str(args.adzuna_max_days_old),
            ]
        )
        # Deeper single-locale passes: fetch into a temp dir, then copy its parquet(s) into
        # the raw dir under a distinct name so the single prep below reads both passes (prep
        # dedupes the overlap from the multi-country fetch). Only for sole-source EU locales
        # (Adzuna is the ONLY source for de/nl, so the shared shallow cap leaves their lanes
        # too thin — ~1k docs -> a near-empty ESCO related lane). EN/FR/others keep the cap.
        countryset = {c.strip().lower() for c in args.adzuna_countries.split(",")}
        deep_passes = {
            "de": (args.adzuna_de_max_pages, args.adzuna_de_max_days_old),
            "nl": (args.adzuna_nl_max_pages, args.adzuna_nl_max_days_old),
            "es": (args.adzuna_es_max_pages, args.adzuna_es_max_days_old),
            "it": (args.adzuna_it_max_pages, args.adzuna_it_max_days_old),
        }
        for loc, (mp, md) in deep_passes.items():
            if loc not in countryset:
                continue
            loc_tmp = adz_dir / f"_raw_{loc}"
            if loc_tmp.exists():
                shutil.rmtree(loc_tmp)
            run(
                [
                    PY,
                    "download/fetch_adzuna.py",
                    "--out-dir",
                    loc_tmp,
                    "--countries",
                    loc,
                    "--max-pages",
                    str(mp),
                    "--max-days-old",
                    str(md),
                ]
            )
            for i, p in enumerate(sorted(loc_tmp.glob("*.parquet"))):
                shutil.copy2(p, adz_raw / f"adzuna-{loc}-{i:04d}.parquet")
            shutil.rmtree(loc_tmp)
        run(
            [
                PY,
                "download/prep_open_apply.py",
                "--raw-dir",
                adz_raw,
                "--out-dir",
                adz_dir,
                "--sample-n",
                "0",
            ]
        )

    # --- Extra ATS tenants (companies not in OpenApply's FINAL slug lists) ---
    adx_dir = ROOT / "jobs_data_ats_extra"
    adx_raw = adx_dir / "raw"
    have_adx_raw = adx_raw.exists() and any(adx_raw.rglob("*.parquet"))
    if args.skip_ats_extra:
        print("[0] --skip-ats-extra: not polling extra ATS tenants", flush=True)
    elif args.skip_download:
        if have_adx_raw:
            print(
                f"[0] --skip-download: reusing existing extra-ATS parquet under {adx_raw}",
                flush=True,
            )
            run(
                [
                    PY,
                    "download/prep_open_apply.py",
                    "--raw-dir",
                    adx_raw,
                    "--out-dir",
                    adx_dir,
                    "--sample-n",
                    "0",
                ]
            )
        else:
            print("[0] --skip-download: no extra-ATS parquet to reuse; skipping", flush=True)
    elif _crawl_ats_extra(adx_raw):
        run(
            [
                PY,
                "download/prep_open_apply.py",
                "--raw-dir",
                adx_raw,
                "--out-dir",
                adx_dir,
                "--sample-n",
                "0",
            ]
        )
    elif have_adx_raw:
        # No fresh tenants this run, but a prior build left a corpus -> clear it so
        # unify doesn't re-add stale extra-ATS jobs.
        shutil.rmtree(adx_dir)
        print("[0] ats-extra: cleared stale corpus (no current tenants)", flush=True)

    # --- Reed (UK), The Muse, Findwork (remote/tech), France Travail (FR) ---
    # Full-description public sources, each cred-gated like Adzuna.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_reed",
        label="Reed",
        have_creds=bool(os.environ.get("REED_API_KEY")),
        fetch_cmd=[
            PY,
            "download/fetch_reed.py",
            "--out-dir",
            ROOT / "jobs_data_reed" / "raw",
            "--max-pages",
            str(args.reed_max_pages),
            "--max-days-old",
            str(args.reed_max_days_old),
        ],
    )
    # The Muse key is optional (the endpoint works keyless), so this corpus is
    # always attempted.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_themuse",
        label="The Muse",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_themuse.py",
            "--out-dir",
            ROOT / "jobs_data_themuse" / "raw",
            "--max-pages",
            str(args.themuse_max_pages),
            "--max-days-old",
            str(args.themuse_max_days_old),
        ],
    )
    _pull_api_source(
        args,
        corpus_dir="jobs_data_findwork",
        label="Findwork",
        have_creds=bool(os.environ.get("FINDWORK_API_KEY")),
        fetch_cmd=[
            PY,
            "download/fetch_findwork.py",
            "--out-dir",
            ROOT / "jobs_data_findwork" / "raw",
            "--max-pages",
            str(args.findwork_max_pages),
            "--max-days-old",
            str(args.findwork_max_days_old),
        ],
    )
    _pull_api_source(
        args,
        corpus_dir="jobs_data_francetravail",
        label="France Travail",
        have_creds=bool(
            os.environ.get("FRANCETRAVAIL_CLIENT_ID") and os.environ.get("FRANCETRAVAIL_SECRET")
        ),
        fetch_cmd=[
            PY,
            "download/fetch_francetravail.py",
            "--out-dir",
            ROOT / "jobs_data_francetravail" / "raw",
            "--publiee-depuis",
            str(args.francetravail_publiee_depuis),
        ],
    )
    # RemoteOK: keyless remote-first board (full HTML descriptions). One endpoint
    # returns the latest ~100 active postings; stable ids let --delta accumulate.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_remoteok",
        label="RemoteOK",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_remoteok.py",
            "--out-dir",
            ROOT / "jobs_data_remoteok" / "raw",
            "--max-days-old",
            str(args.remoteok_max_days_old),
        ],
    )
    # SmartRecruiters + Workable: keyless slug-driven ATSes (full HTML descriptions
    # via a per-posting detail fetch). Companies are NOT in OpenApply's Greenhouse/
    # Lever/Ashby crawl, so they bring net-new inventory. Slug lists were harvested
    # via site-scoped search + verified live (jobs_search_demo/{smartrecruiters,
    # workable}_slugs.txt). Per-company caps bound the nightly detail-fetch cost.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_smartrecruiters",
        label="SmartRecruiters",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_smartrecruiters.py",
            "--out-dir",
            ROOT / "jobs_data_smartrecruiters" / "raw",
            "--slugs-file",
            ROOT / "jobs_search_demo" / "smartrecruiters_slugs.txt",
            "--max-postings-per-company",
            str(args.smartrecruiters_max_postings),
            "--max-days-old",
            str(args.smartrecruiters_max_days_old),
        ],
    )
    _pull_api_source(
        args,
        corpus_dir="jobs_data_workable",
        label="Workable",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_workable.py",
            "--out-dir",
            ROOT / "jobs_data_workable" / "raw",
            "--slugs-file",
            ROOT / "jobs_search_demo" / "workable_slugs.txt",
            "--max-postings-per-company",
            str(args.workable_max_postings),
            "--max-days-old",
            str(args.workable_max_days_old),
        ],
    )
    # Recruitee + Breezy: keyless slug-driven ATSes (SMB-heavy), net-new vs the
    # Greenhouse/Lever/Ashby crawl. Recruitee's offers API carries the full
    # description; Breezy's board JSON does not, so its adapter pulls each
    # position's JobPosting JSON-LD for the body. Slugs verified live; lists in
    # jobs_search_demo/{recruitee,breezy}_slugs.txt.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_recruitee",
        label="Recruitee",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_recruitee.py",
            "--out-dir",
            ROOT / "jobs_data_recruitee" / "raw",
            "--slugs-file",
            ROOT / "jobs_search_demo" / "recruitee_slugs.txt",
            "--max-postings-per-company",
            str(args.recruitee_max_postings),
            "--max-days-old",
            str(args.recruitee_max_days_old),
        ],
    )
    _pull_api_source(
        args,
        corpus_dir="jobs_data_breezy",
        label="Breezy",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_breezy.py",
            "--out-dir",
            ROOT / "jobs_data_breezy" / "raw",
            "--slugs-file",
            ROOT / "jobs_search_demo" / "breezy_slugs.txt",
            "--max-postings-per-company",
            str(args.breezy_max_postings),
            "--max-days-old",
            str(args.breezy_max_days_old),
        ],
    )
    # JobTech (Arbetsförmedlingen): keyless Swedish national board. The JobStream
    # `/stream?date=` endpoint returns every ad changed since a cutoff (full text +
    # pre-attached occupation taxonomy); --max-days-old maps onto that cutoff and
    # stable ids let --delta accumulate. Forced to lang=sv at unify.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_jobtech",
        label="JobTech (Sweden)",
        have_creds=True,
        fetch_cmd=[
            PY,
            "download/fetch_jobtech_se.py",
            "--out-dir",
            ROOT / "jobs_data_jobtech" / "raw",
            "--max-days-old",
            str(args.jobtech_max_days_old),
        ],
    )
    # Jooble: meta-aggregator for extra inventory. Snippet-grade docs, but stage-1
    # unify dedups it against every other corpus so only unique postings land.
    _pull_api_source(
        args,
        corpus_dir="jobs_data_jooble",
        label="Jooble",
        have_creds=bool(os.environ.get("JOOBLE_API_KEY")),
        fetch_cmd=[
            PY,
            "download/fetch_jooble.py",
            "--out-dir",
            ROOT / "jobs_data_jooble" / "raw",
            "--max-pages",
            str(args.jooble_max_pages),
        ],
    )


# ---------------------------------------------------------------------------
# Stage 1: unify (te3-free, 2 corpora, NO vectors)
# ---------------------------------------------------------------------------
def _dedup_sig(rec: dict) -> tuple:
    """Content signature for cross-source dedup: normalized (title, company, primary
    location). Conservative -- including location keeps genuinely-distinct postings of
    the same role in different cities; the goal is to drop a meta-aggregator's re-lists,
    not unique inventory."""
    import re

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    locs = rec.get("locations") or []
    return (
        norm(rec.get("title")),
        (rec.get("source_slug") or "").lower(),
        norm(locs[0] if locs else ""),
    )


def stage_unify(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_ids: list[str] = []
    all_titles: list[str] = []
    all_sources: list[str] = []
    starts: dict[str, int] = {}
    cursor = 0
    # Signatures of every doc kept so far; aggregator corpora dedup against it.
    seen_sigs: set[tuple] = set()

    with open(out / "metadata.jsonl", "w") as meta_out:
        for corpus_dir, tag in CORPORA:
            d = ROOT / corpus_dir
            if not (d / "doc_ids.json").exists():
                # Optional corpus (e.g. Adzuna with no creds) never got prepped; skip it.
                print(f"[1] {corpus_dir}: no doc_ids.json, skipping", flush=True)
                continue
            with open(d / "doc_ids.json") as f:
                ids = json.load(f)
            with open(d / "titles.json") as f:
                titles = json.load(f)
            if len(ids) != len(titles):
                sys.exit(
                    f"[1] size mismatch in {corpus_dir}: {len(ids)} ids vs {len(titles)} titles"
                )
            dedup = tag in DEDUP_AGAINST_PRIOR
            starts[tag] = cursor
            kept = dropped = 0
            with open(d / "metadata.jsonl") as fin:
                for i, line in enumerate(fin):
                    if i >= len(ids):
                        sys.exit(f"[1] {corpus_dir}: more metadata rows than ids ({len(ids)})")
                    rec = json.loads(line)
                    sig = _dedup_sig(rec)
                    if dedup and sig in seen_sigs:
                        dropped += 1
                        continue
                    seen_sigs.add(sig)
                    rec["source_corpus"] = tag
                    # Language tag (en/fr/sv) for the query-language gate + lang facet.
                    # France Travail (fr) and JobTech (sv) are single-language by
                    # construction, so force them and skip the detector; everything else
                    # is detected.
                    src = rec.get("source") or tag
                    if src == "francetravail":
                        rec["lang"] = "fr"
                    elif src == "jobtech":
                        rec["lang"] = "sv"
                    else:
                        rec["lang"] = detect_lang(rec.get("text") or rec.get("title") or "")[0]
                    meta_out.write(json.dumps(rec) + "\n")
                    all_ids.append(ids[i])
                    all_titles.append(titles[i])
                    all_sources.append(tag)
                    kept += 1
            if kept + dropped != len(ids):
                sys.exit(
                    f"[1] {corpus_dir}: processed {kept + dropped} metadata rows vs {len(ids)} ids"
                )
            cursor += kept
            if dedup:
                print(
                    f"[1] {corpus_dir}: {kept:,} unique docs kept "
                    f"({dropped:,} cross-source dupes dropped) starting at {starts[tag]:,}",
                    flush=True,
                )
            else:
                print(f"[1] {corpus_dir}: {kept:,} docs starting at {starts[tag]:,}", flush=True)

    total = cursor
    with open(out / "doc_ids.json", "w") as f:
        json.dump(all_ids, f)
    with open(out / "titles.json", "w") as f:
        json.dump(all_titles, f)
    with open(out / "source_index.json", "w") as f:
        json.dump({"starts": starts, "sources": all_sources}, f)
    print(f"[1] unified {total:,} docs -> {out}  (starts={starts})", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: encode e5-small over unified titles
# ---------------------------------------------------------------------------
def _title_hash(title: str) -> str:
    """Content key for a title's vector. Namespaced by model + doc-prefix so a model
    or prefix change misses every entry, forcing a clean re-encode automatically."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(EMBED_MODEL.encode())
    h.update(b"\x00")
    h.update(EMBED_PREFIX.encode())
    h.update(b"\x00")
    h.update(title.encode("utf-8"))
    return h.hexdigest()


def _run_encode(args, titles_file: str, out_name: str, n_titles: int) -> Path:
    """Encode the titles in `titles_file` (relative to out-dir) into
    `{out_name}.vecs.fp16.npy` via encode_st_catalog, wrapped in the MPS-deadlock
    watchdog. Starts from a clean slate: the encode resume cache (progress.json +
    the .vecs memmap) is POSITIONAL -- it records which chunk indices are done, not
    which titles produced them -- so a prior run's leftovers for this out_name would
    paste stale vectors onto a fresh title set. Within one call, progress.json still
    lets the watchdog skip already-done chunks across an MPS kill/relaunch."""
    import math
    import time

    out = Path(args.out_dir)
    vec = out / f"{out_name}.vecs.fp16.npy"
    progress_path = out / f"{out_name}.progress.json"
    n_chunks = math.ceil(n_titles / EMBED_CHUNK_SIZE)
    for stale in (progress_path, vec):
        if stale.exists():
            stale.unlink()

    def chunks_done() -> int:
        if not progress_path.exists():
            return 0
        try:
            with open(progress_path) as f:
                return len(json.load(f))
        except Exception:
            return 0

    cmd = [
        PY,
        "download/encode_st_catalog.py",
        "--data-dir",
        str(out),
        "--titles-file",
        titles_file,
        "--model",
        EMBED_MODEL,
        "--out-name",
        out_name,
        "--doc-prefix",
        EMBED_PREFIX,
        "--device",
        args.device,
        "--chunk-size",
        str(EMBED_CHUNK_SIZE),
        "--batch-size",
        str(EMBED_BATCH_SIZE),
    ]
    for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
        print(
            f"[2] encode attempt {attempt}/{EMBED_MAX_ATTEMPTS} "
            f"({chunks_done()}/{n_chunks} chunks done, {n_titles:,} titles) on {args.device}",
            flush=True,
        )
        print(f"\n$ {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env={**os.environ})
        last_done = chunks_done()
        last_progress_at = time.time()
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            now_done = chunks_done()
            if now_done != last_done:
                last_done, last_progress_at = now_done, time.time()
            elif time.time() - last_progress_at > EMBED_STALL_SECS:
                print(
                    f"[2] STALL: no chunk completed in {EMBED_STALL_SECS}s at "
                    f"{now_done}/{n_chunks} (likely MPS deadlock); killing + resuming",
                    flush=True,
                )
                proc.kill()
                proc.wait()
                break
            time.sleep(5)
        if ret == 0 and chunks_done() >= n_chunks:
            break
    if not vec.exists() or chunks_done() < n_chunks:
        sys.exit(
            f"[2] encode incomplete after {EMBED_MAX_ATTEMPTS} attempts ({chunks_done()}/{n_chunks})"
        )
    return vec


def stage_encode(args) -> None:
    """Content-addressed DELTA encode. A vector depends only on its title text (with
    the fixed doc-prefix), so a title seen before yields an identical vector. We keep
    a persistent {title-hash -> vector} cache and each night encode ONLY the titles
    new since last run, copying the rest from cache. The crawl is ~95% the same jobs
    day to day, so this turns a ~90min full encode into seconds-to-minutes. The output
    is still the positional `{EMBED_OUT_NAME}.vecs.fp16.npy` (row i == titles[i]) that
    push_docs consumes. Delete the .cache.* files to force a clean rebuild."""
    import hashlib

    import numpy as np

    out = Path(args.out_dir)
    vec = out / f"{EMBED_OUT_NAME}.vecs.fp16.npy"
    fp_path = out / f"{EMBED_OUT_NAME}.titles.sha"
    cache_vec_path = out / f"{EMBED_OUT_NAME}.cache.vecs.fp16.npy"
    cache_key_path = out / f"{EMBED_OUT_NAME}.cache_keys.json"

    raw = (out / "titles.json").read_bytes()
    titles = json.loads(raw)
    n = len(titles)
    fp = hashlib.blake2b(raw, digest_size=16).hexdigest()
    hashes = [_title_hash(t) for t in titles]

    # --- load the content-addressed cache: hash -> row in cache_vecs ---
    cache_vecs: list = []
    cache_idx: dict[str, int] = {}
    if cache_vec_path.exists() and cache_key_path.exists():
        loaded = np.load(cache_vec_path)
        with open(cache_key_path) as f:
            cache_keys = json.load(f)
        cache_vecs = [np.array(loaded[i], dtype=np.float16) for i in range(len(cache_keys))]
        cache_idx = {k: i for i, k in enumerate(cache_keys)}
        print(f"[2] vec cache: {len(cache_idx):,} cached title vectors", flush=True)
    elif vec.exists() and fp_path.exists() and fp_path.read_text().strip() == fp:
        # Bootstrap: a positional vecs file already matches today's titles exactly
        # (same fingerprint), so seed the cache from it for free -- no encode.
        existing = np.load(vec, mmap_mode="r")
        if existing.shape[0] == n:
            for i, h in enumerate(hashes):
                if h not in cache_idx:
                    cache_idx[h] = len(cache_vecs)
                    cache_vecs.append(np.array(existing[i], dtype=np.float16))
            print(
                f"[2] seeded vec cache from matching positional vecs "
                f"({len(cache_idx):,} unique titles, no encode)",
                flush=True,
            )

    # --- which unique titles still need encoding? ---
    need_titles: dict[str, str] = {}
    for h, t in zip(hashes, titles):
        if h not in cache_idx and h not in need_titles:
            need_titles[h] = t

    if need_titles:
        n_new = len(need_titles)
        print(
            f"[2] delta encode: {n_new:,}/{n:,} titles new ({n - n_new:,} reused)",
            flush=True,
        )
        delta_file = "delta_titles.json"
        with open(out / delta_file, "w") as f:
            json.dump(list(need_titles.values()), f)
        delta_vec = _run_encode(args, delta_file, f"{EMBED_OUT_NAME}_delta", n_new)
        new_vecs = np.load(delta_vec)
        if new_vecs.shape[0] != n_new:
            sys.exit(f"[2] delta encode size mismatch: {new_vecs.shape[0]} vs {n_new}")
        for row, h in enumerate(need_titles.keys()):
            cache_idx[h] = len(cache_vecs)
            cache_vecs.append(np.array(new_vecs[row], dtype=np.float16))
    else:
        print(f"[2] delta encode: 0 new titles, all {n:,} reused from cache", flush=True)

    # --- assemble the positional vecs file (row i == titles[i]) push_docs expects ---
    cache_arr = np.asarray(cache_vecs, dtype=np.float16)
    row_for_pos = np.fromiter((cache_idx[h] for h in hashes), dtype=np.int64, count=n)
    np.save(vec, cache_arr[row_for_pos])
    fp_path.write_text(fp)

    # --- prune the cache to today's live titles so it stays bounded, then persist ---
    today = set(hashes)
    keep = [(h, cache_idx[h]) for h in cache_idx if h in today]
    pruned = np.asarray([cache_vecs[i] for _, i in keep], dtype=np.float16)
    np.save(cache_vec_path, pruned)
    with open(cache_key_path, "w") as f:
        json.dump([h for h, _ in keep], f)
    print(f"[2] wrote {vec} ({n:,} rows); cache now {len(keep):,} vectors", flush=True)


# ---------------------------------------------------------------------------
# Stage 2b: snippet passage vectors (content-addressed, like stage_encode)
# ---------------------------------------------------------------------------
def stage_snippet_encode(args) -> None:
    """Pre-encode result-snippet passage vectors so the Space picks snippet passages by
    dot product instead of encoding at query time. Split every staged doc's description
    into candidate passages (snippet_lib), dedup, encode ONLY passages whose vector isn't
    already cached, and write the positional inputs push_docs consumes:
      snippet_passages.vecs.fp16.npy  (row j == j-th unique passage, normalized fp16)
      snippet_doc_rows.json           ({metadata position: [unique-row, ...]})
    The {passage-hash -> vector} cache makes a nightly delta cheap (only NEW postings'
    passages encode). Passages share the title encoder's model + "passage: " prefix, so
    _title_hash keys them in the same namespace. Without this stage push_docs finds no
    artifacts and ships docs WITHOUT snippet_vecs (Space silently falls back to live
    encode). Delete snippet_passages.cache.* to force a clean re-encode."""
    import numpy as np

    out = Path(args.out_dir)
    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    from encode_snippet_vecs import build_passages

    unique, doc_rows, stats = build_passages(str(out))
    with open(out / "snippet_doc_rows.json", "w") as f:
        json.dump(doc_rows, f)
    n_uniq = len(unique)
    print(
        f"[2b] {stats['n_docs']:,} docs -> {n_uniq:,} unique passages "
        f"({stats['total_passages']:,} total)",
        flush=True,
    )

    vec = out / "snippet_passages.vecs.fp16.npy"
    cache_vec_path = out / "snippet_passages.cache.vecs.fp16.npy"
    cache_key_path = out / "snippet_passages.cache_keys.json"
    hashes = [_title_hash(p) for p in unique]

    cache_vecs: list = []
    cache_idx: dict[str, int] = {}
    if cache_vec_path.exists() and cache_key_path.exists():
        loaded = np.load(cache_vec_path)
        with open(cache_key_path) as f:
            cache_keys = json.load(f)
        cache_vecs = [np.array(loaded[i], dtype=np.float16) for i in range(len(cache_keys))]
        cache_idx = {k: i for i, k in enumerate(cache_keys)}
        print(f"[2b] passage cache: {len(cache_idx):,} cached vectors", flush=True)
    elif vec.exists() and (out / "snippet_passages.json").exists():
        # Bootstrap from a prior backfill encode: snippet_passages.json aligns row-for-row
        # with the vecs file, so seed the cache from it (don't re-encode what we paid for).
        with open(out / "snippet_passages.json") as f:
            prior = json.load(f)
        existing = np.load(vec, mmap_mode="r")
        if existing.shape[0] == len(prior):
            for row, p in enumerate(prior):
                h = _title_hash(p)
                if h not in cache_idx:
                    cache_idx[h] = len(cache_vecs)
                    cache_vecs.append(np.array(existing[row], dtype=np.float16))
            print(
                f"[2b] seeded passage cache from backfill ({len(cache_idx):,} vectors)", flush=True
            )

    need: dict[str, str] = {}
    for h, p in zip(hashes, unique):
        if h not in cache_idx and h not in need:
            need[h] = p
    if need:
        n_new = len(need)
        print(f"[2b] delta encode: {n_new:,}/{n_uniq:,} passages new", flush=True)
        delta_file = "snippet_delta.json"
        with open(out / delta_file, "w") as f:
            json.dump(list(need.values()), f)
        delta_vec = _run_encode(args, delta_file, "snippet_passages_delta", n_new)
        new_vecs = np.load(delta_vec)
        if new_vecs.shape[0] != n_new:
            sys.exit(f"[2b] passage delta encode size mismatch: {new_vecs.shape[0]} vs {n_new}")
        for row, h in enumerate(need.keys()):
            cache_idx[h] = len(cache_vecs)
            cache_vecs.append(np.array(new_vecs[row], dtype=np.float16))
    else:
        print(f"[2b] delta encode: 0 new passages, all {n_uniq:,} reused", flush=True)

    # Assemble the positional vecs file (row j == unique[j]) push_docs expects.
    dim = cache_vecs[0].shape[0] if cache_vecs else 384
    if n_uniq:
        cache_arr = np.asarray(cache_vecs, dtype=np.float16)
        row_for_uniq = np.fromiter((cache_idx[h] for h in hashes), dtype=np.int64, count=n_uniq)
        np.save(vec, cache_arr[row_for_uniq])
    else:
        np.save(vec, np.empty((0, dim), dtype=np.float16))

    # Prune the cache to today's passages so it stays bounded, then persist.
    today = set(hashes)
    keep = [(h, cache_idx[h]) for h in cache_idx if h in today]
    pruned = np.asarray([cache_vecs[i] for _, i in keep], dtype=np.float16)
    np.save(cache_vec_path, pruned)
    with open(cache_key_path, "w") as f:
        json.dump([h for h, _ in keep], f)
    print(f"[2b] wrote {vec} ({n_uniq:,} rows); passage cache now {len(keep):,}", flush=True)


# ---------------------------------------------------------------------------
# Stage 3: facets (+ new-slug byproduct)
# ---------------------------------------------------------------------------
def stage_facets(args) -> None:
    sys.path.insert(0, str(ROOT / "jobs_search_demo" / "facets"))
    from heuristics import classify_record  # noqa: E402

    out = Path(args.out_dir)
    facets_out = ROOT / "jobs_search_demo" / "facets" / "facets.jsonl"
    meta_path = out / "metadata.jsonl"

    # Known slugs (for the new-slug byproduct). Prefer the fresh OUT dir, fall
    # back to the legacy unified dir where the label CSV currently lives.
    known_slugs: set[str] = set()
    for base in (out, LEGACY_UNIFIED):
        csv_path = base / INDUSTRY_CSV
        if csv_path.exists():
            import csv

            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    known_slugs.add(r["slug"])
            print(f"[3] loaded {len(known_slugs):,} known slugs from {csv_path}", flush=True)
            # Hand-curated overrides are also "known" -- fold them in so already-labeled
            # slugs don't resurface in the new-slug-to-label byproduct.
            ov_path = base / OVERRIDE_CSV
            if ov_path.exists():
                with open(ov_path) as f:
                    known_slugs.update(r["slug"] for r in csv.DictReader(f))
            break

    new_slugs: set[str] = set()
    role_fams: list[str] = []  # heuristic role_family per idx, for the embedding rescue
    n = 0
    with open(meta_path) as f, open(facets_out, "w") as fo:
        for i, line in enumerate(f):
            rec = json.loads(line)
            facets = classify_record(rec)
            fo.write(json.dumps({"idx": i, **facets}) + "\n")
            role_fams.append(facets.get("role_family") or "other")
            slug = (rec.get("source_slug") or "").strip()
            if slug and slug not in known_slugs:
                new_slugs.add(slug)
            n = i + 1
    print(f"[3] wrote {n:,} facet rows -> {facets_out}", flush=True)

    byproduct = ROOT / "jobs_search_demo" / "facets" / "new_unlabeled_slugs.txt"
    with open(byproduct, "w") as f:
        for s in sorted(new_slugs):
            f.write(s + "\n")
    print(f"[3] {len(new_slugs):,} new unlabeled slugs -> {byproduct}", flush=True)

    _rescue_other_via_embeddings(out, role_fams)
    _rescue_other_via_esco(out, role_fams)
    _rescue_other_via_esco_emb(out, role_fams)
    _rescue_other_via_rome()
    _rescue_other_via_jobtech()
    _rescue_other_via_adzuna(out)


def _rescue_other_via_adzuna(out: Path) -> None:
    """Source-category role_family labels for NON-ENGLISH Adzuna jobs (open-weight,
    no LLM): every Adzuna posting carries a category, whose localized label we cache
    as `department`, mapped straight to a role_family (see classify_other_adzuna /
    adzuna_role_family). English is excluded on purpose -- Adzuna's auto-category is
    noisier than the native English pipeline that already classified those docs.
    Regenerated each refresh; push_docs applies it BEFORE ESCO and only where the
    heuristic == 'other'. Deterministic."""
    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    try:
        from classify_other_adzuna import build
    except Exception as e:  # never abort the refresh on a rescue import failure
        print(f"[3] Adzuna rescue SKIPPED (import failed: {e})", flush=True)
        return

    dest = ROOT / "jobs_search_demo" / "role_family_adzuna_overrides.json"
    try:
        preds = build(str(out / "metadata.jsonl"), str(dest))
    except Exception as e:  # missing metadata -> skip, keep prior file
        print(f"[3] Adzuna rescue SKIPPED ({e})", flush=True)
        return
    print(
        f"[3] Adzuna rescue: {len(preds):,} non-English Adzuna docs labeled -> {dest.name}",
        flush=True,
    )


def _rescue_other_via_jobtech() -> None:
    """Authoritative role_family labels for Swedish JobTech jobs from their
    occupation_field bucket (open-weight, no LLM): every Arbetsförmedlingen ad
    ships with the Swedish occupation taxonomy, whose broad bucket we cache as
    `department` (jobs_data_jobtech/raw), mapped straight to a role_family (see
    classify_other_jobtech / jobtech_role_family). Regenerated each refresh so new
    crawls get labeled; push_docs applies it BEFORE ESCO (source field beats the
    lexical title-match) and only where the heuristic == 'other'. Deterministic."""
    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    try:
        from classify_other_jobtech import build
    except Exception as e:  # never abort the refresh on a rescue import failure
        print(f"[3] JobTech rescue SKIPPED (import failed: {e})", flush=True)
        return

    raw_dir = ROOT / "jobs_data_jobtech" / "raw"
    dest = ROOT / "jobs_search_demo" / "role_family_jobtech_overrides.json"
    try:
        preds = build(str(raw_dir), str(dest))
    except Exception as e:  # missing parquet / pandas -> skip, keep prior file
        print(f"[3] JobTech rescue SKIPPED ({e})", flush=True)
        return
    print(
        f"[3] JobTech rescue: {len(preds):,} Swedish docs labeled -> {dest.name}",
        flush=True,
    )


def _rescue_other_via_rome() -> None:
    """Authoritative role_family labels for France Travail jobs from their ROME
    code (open-weight, no LLM): every FT offer carries a ROME 4.0 occupation code
    assigned by France Travail (cached in jobs_data_francetravail/raw), mapped
    straight to a role_family (see classify_other_rome / rome_role_family).
    Regenerated each refresh so new FT crawls get labeled; push_docs applies it
    BEFORE ESCO (authoritative source code beats the lexical title-match) and only
    where the heuristic == 'other'. Deterministic -> reproducible."""
    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    try:
        from classify_other_rome import build
    except Exception as e:  # never abort the refresh on a rescue import failure
        print(f"[3] ROME rescue SKIPPED (import failed: {e})", flush=True)
        return

    raw_dir = ROOT / "jobs_data_francetravail" / "raw"
    dest = ROOT / "jobs_search_demo" / "role_family_rome_overrides.json"
    try:
        preds = build(str(raw_dir), str(dest))
    except Exception as e:  # missing parquet / pandas -> skip, keep prior file
        print(f"[3] ROME rescue SKIPPED ({e})", flush=True)
        return
    print(
        f"[3] ROME rescue: {len(preds):,} France Travail docs labeled -> {dest.name}",
        flush=True,
    )


def _rescue_other_via_esco(out: Path, role_fams: list[str]) -> None:
    """Open-weight multilingual role_family labels for the non-English 'other'
    residual (no LLM): each fr/sv/de/nl/es/it title is matched to an ESCO occupation
    in its own language and the occupation's ISCO-08 code maps to a role_family (see
    classify_other_esco / isco_role_family). Regenerated each refresh so new docs get
    labeled; push_docs applies it (after emb/llm) only where the heuristic == 'other'.
    Deterministic -> reproducible on an unchanged corpus."""
    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    try:
        from classify_other_esco import rescue
    except Exception as e:  # never abort the refresh on a rescue import failure
        print(f"[3] ESCO rescue SKIPPED (import failed: {e})", flush=True)
        return

    preds = rescue(out / "metadata.jsonl", role_fams)
    dest = ROOT / "jobs_search_demo" / "role_family_esco_overrides.json"
    with open(dest, "w") as f:
        json.dump(preds, f, indent=0, sort_keys=True, ensure_ascii=False)
    print(
        f"[3] ESCO rescue: {len(preds):,} non-English 'other' docs labeled -> {dest.name}",
        flush=True,
    )


def _rescue_other_via_esco_emb(out: Path, role_fams: list[str]) -> None:
    """Multilingual-embedding ESCO role_family labels for the es/it 'other' residual
    (open-weight, no LLM): each title is matched to its nearest ESCO occupation label
    by multilingual-e5 cosine, gated on cosine + vote-share, and the occupation's
    ISCO-08 code maps to a role_family (see classify_other_esco_emb / isco_role_family).
    The encode is OFFLINE (match step only) -- the served retrieval model is unchanged.
    Scoped to es/it/de/nl (the audited non-English residual locales; fr/sv are already
    covered by authoritative ROME/JobTech); push_docs applies it with
    the LOWEST precedence (after every authoritative source + lexical ESCO) only where
    the heuristic == 'other'. Deterministic on a fixed corpus + model -> reproducible."""
    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    try:
        from classify_other_esco_emb import rescue as esco_emb_rescue
    except Exception as e:  # sentence-transformers absent -> skip, don't abort refresh
        print(f"[3] emb-ESCO rescue SKIPPED (import failed: {e})", flush=True)
        return

    preds = esco_emb_rescue(out / "metadata.jsonl", role_fams, langs=("es", "it", "de", "nl"))
    dest = ROOT / "jobs_search_demo" / "role_family_esco_emb_overrides.json"
    with open(dest, "w") as f:
        json.dump(preds, f, indent=0, sort_keys=True, ensure_ascii=False)
    print(
        f"[3] emb-ESCO rescue: {len(preds):,} es/it/de/nl 'other' docs labeled -> {dest.name}",
        flush=True,
    )


def _rescue_other_via_embeddings(out: Path, role_fams: list[str]) -> None:
    """Embedding-based role_family labels for the 'other' residual, regenerated
    each refresh so NEW docs get rescued (the override file is otherwise a frozen
    snapshot). Runs the ensemble-gated e5 kNN of classify_other_emb over the full
    catalog vectors (stage 2 output, positionally aligned to doc_ids/metadata) and
    overwrites role_family_emb_overrides.json, which push_docs applies at index
    time over the heuristic label (only where the heuristic == 'other'). Reproduces
    the prior result byte-for-byte on an unchanged corpus; grows as new docs land."""
    import numpy as np

    sys.path.insert(0, str(ROOT / "jobs_search_demo"))
    try:
        from classify_other_emb import _norm, classify_other, load_depts, load_text
    except Exception as e:  # faiss/sentence-transformers absent -> skip, don't abort refresh
        print(f"[3] embedding rescue SKIPPED (import failed: {e})", flush=True)
        return

    vec_path = out / f"{EMBED_OUT_NAME}.vecs.fp16.npy"
    ids_path = out / "doc_ids.json"
    if not vec_path.exists() or not ids_path.exists():
        print("[3] embedding rescue SKIPPED (vectors/doc_ids absent — run stage 2)", flush=True)
        return

    with open(ids_path) as f:
        ids = json.load(f)
    V = _norm(np.load(vec_path))
    if not (len(ids) == V.shape[0] == len(role_fams)):
        print(
            f"[3] embedding rescue SKIPPED (alignment mismatch: ids={len(ids):,} "
            f"V={V.shape[0]:,} fams={len(role_fams):,})",
            flush=True,
        )
        return

    y = np.array(role_fams)
    txt = load_text(out / "metadata.jsonl")
    depts = load_depts(out / "metadata.jsonl")
    preds, _dropped, cnt = classify_other(ids, V, y, txt, depts=depts)
    dest = ROOT / "jobs_search_demo" / "role_family_emb_overrides.json"
    # indent=0 + sort_keys: one key per line, stable order -> reviewable line-level
    # diffs as the override set grows refresh to refresh.
    with open(dest, "w") as f:
        json.dump(preds, f, indent=0, sort_keys=True)
    print(
        f"[3] embedding rescue: {len(preds):,} 'other' docs labeled -> {dest.name} (dropped {cnt})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Stage 4: Solr build (live mutation)
# ---------------------------------------------------------------------------
def _solr_env() -> dict:
    return {"JAVA_HOME": JAVA_HOME}  # force Java 21; run.sh's :- default is unsafe


def stage_solr(args) -> None:
    import time
    import urllib.request

    # Absolute: push_docs runs with cwd=demo, so a relative JOBS_STAGE would
    # wrongly resolve under jobs_search_demo/ instead of the repo root.
    out = Path(args.out_dir).resolve()
    demo = ROOT / "jobs_search_demo"
    solr_bin = "/opt/homebrew/bin/solr"

    if getattr(args, "delta", False):
        return _stage_solr_delta(args, out, demo, solr_bin)

    # The fresh OUT dir needs the slug->industry CSVs that push_docs reads from STAGE.
    _seed_label_csvs(out, "4")
    dst_csv = out / INDUSTRY_CSV

    # Start Solr (idempotent).
    def solr_up() -> bool:
        try:
            with urllib.request.urlopen(f"{SOLR_URL}/solr/admin/info/system", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    if not solr_up():
        run(
            [solr_bin, "start", "--user-managed", "--solr-home", SOLR_HOME, "-p", "8983"],
            env=_solr_env(),
        )
        for _ in range(60):
            if solr_up():
                break
            time.sleep(1)
        else:
            sys.exit("[4] Solr did not come up on 8983")

    # (Re)create the core with conf materialized INSIDE the instance dir. Solr
    # auto-discovers any existing core at startup, so deleting its files while it is
    # loaded leaves it half-alive and CREATE fails with "already exists". UNLOAD it
    # from the running instance FIRST, then wipe the dir, then CREATE.
    #
    # We do NOT use configSet=_default: that records `configSet=_default` in
    # core.properties (referenced BY NAME) instead of copying conf/ into the core,
    # so the deploy tar (which contains only the core dir) is missing its config and
    # the Space — which has no configsets/_default — fails to load the core and every
    # /select 500s. Seed conf/ into jobs/conf instead, so the tar is self-contained.
    def core_loaded() -> bool:
        try:
            with urllib.request.urlopen(
                f"{SOLR_URL}/solr/admin/cores?action=STATUS&core={CORE}", timeout=10
            ) as r:
                return bool(json.load(r).get("status", {}).get(CORE))
        except Exception:
            return False

    if core_loaded():
        run(
            [
                "curl",
                "-sS",
                f"{SOLR_URL}/solr/admin/cores?action=UNLOAD&core={CORE}"
                "&deleteIndex=true&deleteDataDir=true&deleteInstanceDir=true",
            ],
            env=_solr_env(),
        )
    instance_dir = SOLR_HOME / CORE
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    shipped_conf = Path("/opt/homebrew/opt/solr/server/solr/configsets/_default/conf")
    if not shipped_conf.exists():
        sys.exit(f"[4] no _default conf to seed from {shipped_conf}")
    shutil.copytree(shipped_conf, instance_dir / "conf")
    print(f"[4] seeded core conf -> {instance_dir / 'conf'}", flush=True)
    create_url = (
        f"{SOLR_URL}/solr/admin/cores?action=CREATE&name={CORE}&instanceDir={CORE}&dataDir=data"
    )
    # --fail-with-body: HTTP 4xx/5xx -> non-zero exit (plain `curl -sS` swallows them,
    # which is how a failed CREATE previously slipped through into a broken push).
    run(["curl", "-sS", "--fail-with-body", create_url], env=_solr_env())

    # Schema: base + facets + the industry field that BOTH schema scripts omit.
    run(["bash", "configure_schema.sh"], cwd=demo)
    run(["bash", "add_facet_fields.sh"], cwd=demo)
    industry_field = json.dumps(
        {
            "add-field": [
                {
                    "name": "industry",
                    "type": "string",
                    "indexed": True,
                    "stored": True,
                    "multiValued": False,
                }
            ]
        }
    )
    run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            f"{SOLR_URL}/solr/{CORE}/schema",
            "--data-binary",
            industry_field,
        ]
    )

    # Quality check (report-only): sample the confidence-gated industry labels so a
    # contaminated bucket like the old education_higher (~75% wrong) is caught before it
    # ships again. Non-fatal by design — job freshness shouldn't hinge on a facet-quality
    # blip — but the sample TSV lands in the deploy dir and any RED FLAG is logged loudly.
    # (Run qc_industry_labels.py standalone for the strict, exit-code gate.)
    qc_out = out / "qc_industry_sample.tsv"
    qc = subprocess.run(
        [
            PY,
            "qc_industry_labels.py",
            "--labels",
            str(dst_csv),
            "--meta",
            str(out / "metadata.jsonl"),
            "--out",
            str(qc_out),
        ],
        cwd=str(demo),
        env={**os.environ},
        check=False,
    )
    if qc.returncode != 0:
        print(f"[4] *** industry-label QC raised red flags — review {qc_out} ***", flush=True)

    # Snippet passage vectors are a push precondition: produce them now (right before the
    # push) so push_docs attaches snippet_vecs. Done here rather than as a numbered stage
    # so the integer --from/--to-stage range (and the >=4 live-mutation guard) is unchanged.
    stage_snippet_encode(args)

    # push_docs reads from STAGE (hardcoded to unified_jobs). If we built into a
    # different dir, point it there via env override.
    push_env = {"JOBS_STAGE": str(out)} if str(out) != str(LEGACY_UNIFIED) else {}
    if push_env:
        print(
            "[4] NOTE: push_docs STAGE is hardcoded; set JOBS_STAGE handling or run with --out-dir unified_jobs",
            flush=True,
        )
    run([PY, "push_docs.py"], cwd=demo, env=push_env)
    run(["curl", "-sS", "--fail-with-body", f"{SOLR_URL}/solr/{CORE}/update?commit=true"])

    with urllib.request.urlopen(f"{SOLR_URL}/solr/{CORE}/select?q=*:*&rows=0", timeout=10) as r:
        n = json.load(r)["response"]["numFound"]
    with open(out / "metadata.jsonl") as mf:
        expected = sum(1 for _ in mf)
    print(f"[4] indexed numFound={n:,} (expected {expected:,})", flush=True)
    # Hard gate: a short/empty index must NOT reach tar/upload/deploy.
    if n < 0.99 * expected:
        sys.exit(f"[4] ABORT: indexed {n:,} < 99% of expected {expected:,}")

    _regen_fr_suggestions(demo)


def _stage_solr_delta(args, out: Path, demo: Path, solr_bin: str) -> None:
    """Incremental stage 4: diff the new build's stable ids against what's already
    in the persistent core, then push only the ADDED postings and delete the CLOSED
    ones. No wipe — so untouched Lucene segments stay byte-identical and the stage-6
    Xet upload dedups them (measured: ~4MB / ~5s for a daily delta vs ~900MB full).

    NOTE: a posting that is re-listed under the same doc id keeps its id, so an
    in-place CONTENT edit (e.g. a reworded description) is NOT re-pushed here — the
    periodic full rebuild (--no-delta) reconciles any such drift."""
    import time
    import urllib.request as u

    _seed_label_csvs(out, "4d")

    def solr_up() -> bool:
        try:
            with u.urlopen(f"{SOLR_URL}/solr/admin/info/system", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    if not solr_up():
        run(
            [solr_bin, "start", "--user-managed", "--solr-home", SOLR_HOME, "-p", "8983"],
            env=_solr_env(),
        )
        for _ in range(60):
            if solr_up():
                break
            time.sleep(1)
        else:
            sys.exit("[4d] Solr did not come up on 8983")

    def core_loaded() -> bool:
        try:
            with u.urlopen(
                f"{SOLR_URL}/solr/admin/cores?action=STATUS&core={CORE}", timeout=10
            ) as r:
                return bool(json.load(r).get("status", {}).get(CORE))
        except Exception:
            return False

    # A missing core => create it fresh (+schema) WITHOUT a wipe; the diff below then
    # resolves to "add everything", i.e. a self-healing full push on the first run.
    if not core_loaded():
        print("[4d] core absent — creating fresh (delta resolves to a full push)", flush=True)
        instance_dir = SOLR_HOME / CORE
        shipped_conf = Path("/opt/homebrew/opt/solr/server/solr/configsets/_default/conf")
        if not shipped_conf.exists():
            sys.exit(f"[4d] no _default conf to seed from {shipped_conf}")
        if instance_dir.exists():
            shutil.rmtree(instance_dir)
        shutil.copytree(shipped_conf, instance_dir / "conf")
        create_url = (
            f"{SOLR_URL}/solr/admin/cores?action=CREATE&name={CORE}&instanceDir={CORE}&dataDir=data"
        )
        run(["curl", "-sS", "--fail-with-body", create_url], env=_solr_env())
        run(["bash", "configure_schema.sh"], cwd=demo)
        run(["bash", "add_facet_fields.sh"], cwd=demo)
        industry_field = json.dumps(
            {
                "add-field": [
                    {
                        "name": "industry",
                        "type": "string",
                        "indexed": True,
                        "stored": True,
                        "multiValued": False,
                    }
                ]
            }
        )
        run(
            [
                "curl",
                "-sS",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                f"{SOLR_URL}/solr/{CORE}/schema",
                "--data-binary",
                industry_field,
            ]
        )

    # New build's stable ids (same scheme push_docs uses), aligned with row position.
    sys.path.insert(0, str(demo))
    from push_docs import stable_id  # noqa: E402

    with open(out / "doc_ids.json") as f:
        doc_ids = [str(x) for x in json.load(f)]
    new_ids = [stable_id(d) for d in doc_ids]
    new_set = set(new_ids)
    if len(new_set) != len(new_ids):
        sys.exit("[4d] stable_id collision in new build — widen digest_size")

    # Existing ids in the live core (Solr is the source of truth).
    count_url = f"{SOLR_URL}/solr/{CORE}/select?q=*:*&rows=0&wt=json"
    with u.urlopen(count_url, timeout=30) as r:
        n_existing = json.load(r)["response"]["numFound"]
    existing: set[int] = set()
    if n_existing:
        with u.urlopen(
            f"{SOLR_URL}/solr/{CORE}/select?q=*:*&fl=id&rows={n_existing + 1000}&wt=json",
            timeout=180,
        ) as r:
            existing = {int(d["id"]) for d in json.load(r)["response"]["docs"]}
    add_positions = [i for i, sid in enumerate(new_ids) if sid not in existing]
    del_ids = sorted(existing - new_set)
    print(
        f"[4d] existing={len(existing):,} new={len(new_set):,} -> "
        f"add={len(add_positions):,} delete={len(del_ids):,}",
        flush=True,
    )

    if add_positions:
        # Produce snippet passage vectors before the push so the added docs carry
        # snippet_vecs. Content-addressed, so a delta only encodes the new postings'
        # passages (cache hits cover the rest).
        stage_snippet_encode(args)
        pos_file = out / "_delta_positions.json"
        with open(pos_file, "w") as f:
            json.dump(add_positions, f)
        run(
            [PY, "push_docs.py"],
            cwd=demo,
            env={"JOBS_STAGE": str(out), "JOBS_NO_CLEAR": "1", "JOBS_POSITIONS": str(pos_file)},
        )
        pos_file.unlink()

    for j in range(0, len(del_ids), 1000):
        body = json.dumps({"delete": [str(x) for x in del_ids[j : j + 1000]]}).encode()
        req = u.Request(
            f"{SOLR_URL}/solr/{CORE}/update",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        u.urlopen(req, timeout=120).read()

    run(["curl", "-sS", "--fail-with-body", f"{SOLR_URL}/solr/{CORE}/update?commit=true"])
    with u.urlopen(count_url, timeout=30) as r:
        n_final = json.load(r)["response"]["numFound"]
    expected = len(new_set)
    print(f"[4d] post-delta numFound={n_final:,} (expected {expected:,})", flush=True)
    if n_final != expected:
        sys.exit(f"[4d] ABORT: post-delta numFound {n_final:,} != expected {expected:,}")

    _regen_fr_suggestions(demo)


def _regen_fr_suggestions(demo: Path) -> None:
    """Regenerate the per-language serve-time suggestion bundles from the freshly
    built live Solr corpus so they track current inventory (they are otherwise
    frozen snapshots that drift as postings turn over):
      * space/fr_roles.json   (mine_fr_roles.py)   — French role vocab for
        autocomplete + resume matching.
      * space/fr_related.json (build_fr_related.py) — the grounded ROME
        related-search lane, whose display labels must EXIST in the live corpus.
      * space/sv_roles.json   (mine_sv_roles.py)   — Swedish (JobTech) role vocab
      * space/sv_related.json (build_sv_related.py) — the grounded ESCO related-search
        for the Swedish autocomplete tier.
      * space/de_roles.json   (mine_de_roles.py)   — German (Adzuna DE) role vocab
        for the German autocomplete tier.
      * space/de_related.json (build_de_related.py) — the grounded ESCO related-search
        lane, whose display labels must EXIST in the live corpus.
      * space/es_roles.json   (mine_es_roles.py)   — Spanish (Adzuna ES) role vocab
      * space/es_related.json (build_es_related.py) — the grounded ESCO related-search
      * space/nl_roles.json   (mine_nl_roles.py)   — Dutch (Adzuna NL) role vocab
        for the Dutch autocomplete tier.
      * space/nl_related.json (build_nl_related.py) — the grounded ESCO related-search
        lane, whose display labels must EXIST in the live corpus.
      * space/it_roles.json   (mine_it_roles.py)   — Italian (Adzuna IT) role vocab
      * space/it_related.json (build_it_related.py) — the grounded ESCO related-search
        lane, whose display labels must EXIST in the live corpus.
    All read Solr :8983, so this runs AFTER the stage-4 commit and BEFORE the
    stage-7 deploy uploads them. Non-fatal: a failure keeps the last-good bundle
    rather than aborting the index refresh. build_fr_related needs a one-time
    ROME open-data download (cached at .rome_opendata.zip); build_de_related/build_nl_related
    read the harvested ESCO backbone (.esco_records.jsonl); the miners write cwd-relative
    paths, hence cwd=demo for all."""
    for script in (
        "mine_fr_roles.py",
        "build_fr_related.py",
        "mine_sv_roles.py",
        "build_sv_related.py",
        "mine_de_roles.py",
        "build_de_related.py",
        "mine_nl_roles.py",
        "build_nl_related.py",
        "mine_es_roles.py",
        "build_es_related.py",
        "mine_it_roles.py",
        "build_it_related.py",
    ):
        try:
            run([PY, script], cwd=demo)
            print(f"[4] regenerated suggestion bundle via {script}", flush=True)
        except Exception as e:  # noqa: BLE001 — degrade to last-good, never abort refresh
            print(f"[4] WARNING: {script} failed ({e}); keeping last-good bundle", flush=True)


# ---------------------------------------------------------------------------
# Stage 5: tar
# ---------------------------------------------------------------------------
def stage_tar(args) -> None:
    tar_path = ROOT / "jobs_search_demo" / "solr_jobs_core.tar"
    run(
        ["tar", "--no-xattrs", "-cf", tar_path, "-C", SOLR_HOME, CORE],
        env={"COPYFILE_DISABLE": "1"},
    )
    print(f"[5] wrote {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)", flush=True)


# ---------------------------------------------------------------------------
# Stage 6: upload to HF dataset
# ---------------------------------------------------------------------------
def stage_upload(args) -> None:
    from huggingface_hub import HfApi

    tar_path = ROOT / "jobs_search_demo" / "solr_jobs_core.tar"
    print(
        "[6] uploading to dtunkelang/jobs-demo (expect a 30-45 min stall near 99-100%)...",
        flush=True,
    )
    HfApi().upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo="solr_index/solr_jobs_core.tar",
        repo_id="dtunkelang/jobs-demo",
        repo_type="dataset",
    )
    print("[6] upload complete", flush=True)


# ---------------------------------------------------------------------------
# Stage 7: deploy app + restart Space
# ---------------------------------------------------------------------------
def stage_deploy(args) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    space_id = "dtunkelang/jobs-search"
    space_dir = ROOT / "jobs_search_demo" / "space"
    # lang_detect.py is canonical at the demo root (imported here by stage_unify for
    # index-time lang tagging AND by the miners); the Space serves its OWN copy under
    # space/. Sync it before upload so the served query-gate can't silently drift behind
    # the index tagging (this exact drift once shipped an en+fr-only gate while the index
    # already carried sv/de docs).
    shutil.copy2(ROOT / "jobs_search_demo" / "lang_detect.py", space_dir / "lang_detect.py")
    # The merged profile-match lane needs resume_match_lib.py (imported by app.py)
    # and pypdf (requirements.txt) on the Space — and the Dockerfile must COPY the
    # lib into the image, so push the Dockerfile too (not just app.py).
    for fname in (
        "app.py",
        "README.md",
        "snippet_lib.py",
        "resume_match_lib.py",
        "maps_svg.py",
        "lang_detect.py",
        "requirements.txt",
        "Dockerfile",
        "entrypoint.sh",
        "suggest_lib.py",
        "employer_names.json",
        "employer_names_extracted.json",
        "role_vocab.json",
        "role_vocab_emb.npy",
        "fr_roles.json",
        "fr_related.json",
        "sv_roles.json",
        "sv_related.json",
        "de_roles.json",
        "de_related.json",
        "nl_roles.json",
        "nl_related.json",
        "es_roles.json",
        "es_related.json",
        "it_roles.json",
        "it_related.json",
    ):
        api.upload_file(
            path_or_fileobj=str(space_dir / fname),
            path_in_repo=fname,
            repo_id=space_id,
            repo_type="space",
        )
    api.restart_space(space_id)
    print(
        f"[7] redeployed app.py + restarted {space_id}; verify get_space_runtime().stage==RUNNING",
        flush=True,
    )


STAGES = [
    ("pull", stage_pull),
    ("unify", stage_unify),
    ("encode", stage_encode),
    ("facets", stage_facets),
    ("solr", stage_solr),
    ("tar", stage_tar),
    ("upload", stage_upload),
    ("deploy", stage_deploy),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from-stage", type=int, default=0)
    ap.add_argument(
        "--to-stage", type=int, default=3, help="inclusive; default 3 (dry-run data pipeline)"
    )
    ap.add_argument(
        # Default to the SAME dir the nightly (daily_jobs_refresh.sh) uses, so a manual
        # refresh reuses its warm content-addressed encode cache (a cold dir re-encodes
        # all ~330k titles on MPS for nothing) AND stays catalog-consistent with the live
        # core the nightly maintains. Use an explicit --out-dir only for isolated experiments.
        "--out-dir",
        default=str(ROOT / "unified_jobs_daily"),
        help="unified catalog output dir (default matches the nightly's warm-cache dir)",
    )
    ap.add_argument(
        "--openapply-source",
        choices=["hf", "crawl"],
        default="crawl",
        help="crawl (default): run the crawler directly (~25-30min, same-day "
        "fresh); hf: fast download of maintainer's HF snapshot (may lag ~1wk)",
    )
    ap.add_argument(
        "--crawl-workers", type=int, default=16, help="oa_adapter --workers (crawl source)"
    )
    ap.add_argument(
        "--snapshot-date",
        default="latest",
        help="hf source only: daily partition 'latest' (freshest full catalog), 'all', or YYYY-MM-DD",
    )
    ap.add_argument("--openapply-sample-n", type=int, default=0, help="0 = keep all (post-dedup)")
    ap.add_argument(
        "--adzuna-countries",
        # English-speaking countries + France + Germany + Netherlands + Spain + Italy: every
        # locale the index handles (English e5, the French lang-gate/ROME related-lane, and
        # the German/Dutch/Spanish/Italian lang-gate/ESCO related-lanes). The remaining Adzuna
        # locales (pl/...) are omitted on purpose — no lang handling exists for them yet, so
        # they'd contaminate the index.
        default="us,ca,gb,au,nz,in,sg,za,fr,de,nl,es,it",
        help="comma-separated Adzuna country codes (us,gb,ca,...); needs ADZUNA_APP_ID/KEY",
    )
    ap.add_argument(
        "--adzuna-max-pages", type=int, default=20, help="Adzuna pages per country (50 jobs/page)"
    )
    ap.add_argument(
        "--adzuna-max-days-old", type=int, default=7, help="Adzuna: only postings newer than N days"
    )
    # Germany gets a dedicated deeper pass: Adzuna is the SOLE German source (unlike
    # English, which has many), so the shared per-country cap (--adzuna-max-pages) leaves
    # the German lane too thin (~1k docs -> a near-empty ESCO related lane). A second
    # de-only fetch with a larger page/day budget brings it to ~5k. EN/FR are unaffected.
    ap.add_argument(
        "--adzuna-de-max-pages", type=int, default=120, help="extra de-only Adzuna pass: pages"
    )
    ap.add_argument(
        "--adzuna-de-max-days-old", type=int, default=14, help="extra de-only Adzuna pass: days"
    )
    # Netherlands gets the same dedicated deeper pass for the same reason (Adzuna is the
    # SOLE Dutch source, so the shared cap leaves the Dutch ESCO related lane too thin).
    ap.add_argument(
        "--adzuna-nl-max-pages", type=int, default=120, help="extra nl-only Adzuna pass: pages"
    )
    ap.add_argument(
        "--adzuna-nl-max-days-old", type=int, default=14, help="extra nl-only Adzuna pass: days"
    )
    # Spain gets the same dedicated deeper pass for the same reason (Adzuna is the SOLE
    # Spanish source, so the shared cap leaves the Spanish ESCO related lane too thin).
    ap.add_argument(
        "--adzuna-es-max-pages", type=int, default=120, help="extra es-only Adzuna pass: pages"
    )
    ap.add_argument(
        "--adzuna-es-max-days-old", type=int, default=14, help="extra es-only Adzuna pass: days"
    )
    # Italy gets the same dedicated deeper pass for the same reason (Adzuna is the SOLE
    # Italian source, so the shared cap leaves the Italian ESCO related lane too thin).
    ap.add_argument(
        "--adzuna-it-max-pages", type=int, default=120, help="extra it-only Adzuna pass: pages"
    )
    ap.add_argument(
        "--adzuna-it-max-days-old", type=int, default=14, help="extra it-only Adzuna pass: days"
    )
    # Additional public sources (cred-gated). Modest defaults keep nightly --delta
    # runs cheap; raise per source for a fuller backfill.
    ap.add_argument("--reed-max-pages", type=int, default=20, help="Reed pages (100 jobs/page)")
    ap.add_argument(
        "--reed-max-days-old", type=int, default=14, help="Reed: only postings newer than N days"
    )
    ap.add_argument(
        "--themuse-max-pages", type=int, default=50, help="The Muse pages (20 jobs/page)"
    )
    ap.add_argument(
        "--themuse-max-days-old",
        type=int,
        default=14,
        help="The Muse: only postings newer than N days",
    )
    ap.add_argument("--findwork-max-pages", type=int, default=20, help="Findwork pages")
    ap.add_argument(
        "--findwork-max-days-old",
        type=int,
        default=14,
        help="Findwork: only postings newer than N days",
    )
    ap.add_argument(
        "--francetravail-publiee-depuis",
        type=int,
        default=7,
        choices=[1, 3, 7, 14, 31],
        help="France Travail: only offers published within the last N days",
    )
    ap.add_argument(
        "--remoteok-max-days-old",
        type=int,
        default=30,
        help="RemoteOK: only postings newer than N days",
    )
    ap.add_argument(
        "--smartrecruiters-max-postings",
        type=int,
        default=300,
        help="SmartRecruiters: max postings per company (0 = all; caps detail-fetch cost)",
    )
    ap.add_argument(
        "--smartrecruiters-max-days-old",
        type=int,
        default=30,
        help="SmartRecruiters: only postings newer than N days",
    )
    ap.add_argument(
        "--workable-max-postings",
        type=int,
        default=0,
        help="Workable: max postings per company (0 = all)",
    )
    ap.add_argument(
        "--workable-max-days-old",
        type=int,
        default=30,
        help="Workable: only postings newer than N days",
    )
    ap.add_argument(
        "--recruitee-max-postings",
        type=int,
        default=0,
        help="Recruitee: max postings per company (0 = all)",
    )
    ap.add_argument(
        "--recruitee-max-days-old",
        type=int,
        default=30,
        help="Recruitee: only postings newer than N days",
    )
    ap.add_argument(
        "--breezy-max-postings",
        type=int,
        default=0,
        help="Breezy: max postings per company (0 = all)",
    )
    ap.add_argument(
        "--breezy-max-days-old",
        type=int,
        default=30,
        help="Breezy: only postings newer than N days",
    )
    ap.add_argument(
        "--jobtech-max-days-old",
        type=int,
        default=7,
        help="JobTech (Sweden): pull the JobStream delta covering the last N days",
    )
    ap.add_argument(
        "--jooble-max-pages",
        type=int,
        default=5,
        help="Jooble result pages per keyword in job_search_queries.txt",
    )
    ap.add_argument(
        "--skip-ats-extra",
        action="store_true",
        help="don't poll the extra Greenhouse/Lever/Ashby tenants in "
        "jobs_search_demo/extra_ats_slugs/cc_*_EXTRA.txt",
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-download", action="store_true", help="reuse existing raw parquet")
    ap.add_argument("--no-dry-run", action="store_true", help="allow live-mutation stages 4-7")
    ap.add_argument(
        "--delta",
        action="store_true",
        help="stage 4: incrementally upsert into the persistent core (add new + delete "
        "closed postings) instead of wipe+full-reindex. Keeps Lucene segments byte-stable "
        "so the stage-6 Xet upload dedups (~MB/secs vs full ~900MB). Requires the core to "
        "already exist from a prior full build (a missing core falls back to a full push).",
    )
    args = ap.parse_args()

    if args.to_stage >= 4 and not args.no_dry_run:
        sys.exit("refusing stages >=4 without --no-dry-run (those mutate the live demo)")

    for i in range(args.from_stage, args.to_stage + 1):
        name, fn = STAGES[i]
        print(f"\n{'=' * 70}\n=== STAGE {i}: {name}\n{'=' * 70}", flush=True)
        fn(args)
    print(f"\nDONE stages {args.from_stage}-{args.to_stage}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
