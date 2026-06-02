#!/usr/bin/env python3
"""Fetch currently-posted jobs from the France Travail (ex-Pole Emploi) API.

France Travail is France's national public employment service; its "Offres
d'emploi v2" API exposes the full live national job inventory (~590k postings),
so it brings non-US/French-language inventory that the OpenApply crawler
(Greenhouse/Lever/Ashby), USAJOBS (federal), Adzuna, SmartRecruiters and Workable
do not cover. The search response carries the full description inline, so unlike
the ATS adapters this needs no per-posting detail fetch.

Requires free credentials (OAuth2 client_credentials). Register an application at
https://francetravail.io, subscribe it to "Offres d'emploi v2", then export:
  FRANCETRAVAIL_CLIENT_ID='PAR_...'
  FRANCETRAVAIL_SECRET='...'

The search endpoint hard-caps any single query at 3,000 results, so this script
partitions by departement (and bounds recency via --publiee-depuis) to cover the
whole country while staying under the cap. The OAuth token (~25 min TTL) is
refreshed automatically mid-run.

Writes parquet files whose schema matches download/fetch_adzuna.py / what
prep_open_apply.py expects (id, source_slug, title, description_html, department,
employment_type, remote, locations, salary_min, salary_max, salary_currency,
posted_at, source).

Usage:
  .venv/bin/python download/fetch_francetravail.py \\
      --out-dir jobs_data_francetravail/raw \\
      --publiee-depuis 7
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
SCOPE = "api_offresdemploiv2 o2dsoffre"
PAGE = 150  # API max results_per_page
RANGE_CAP = 3000  # API refuses range offsets at/over this; partitioning keeps us under it
SLUG_RE = re.compile(r"[^a-z0-9]+")
TAG_RE = re.compile(r"<[^>]+>")
# 01-95 metropolitan (20 -> 2A/2B Corsica) + the overseas departements.
DEPARTEMENTS = (
    [f"{d:02d}" for d in range(1, 96) if d != 20]
    + ["2A", "2B"]
    + ["971", "972", "973", "974", "976"]
)


def slugify(s: str) -> str:
    return SLUG_RE.sub("-", (s or "").lower()).strip("-")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s or "").strip()


class Token:
    """Lazily fetched client_credentials token, refreshed before it expires."""

    def __init__(self, cid: str, secret: str):
        self.cid, self.secret = cid, secret
        self._token = ""
        self._expires_at = 0.0

    def get(self) -> str:
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        body = urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.cid,
                "client_secret": self.secret,
                "scope": SCOPE,
            }
        ).encode()
        req = Request(
            TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        self._token = d["access_token"]
        self._expires_at = time.time() + int(d.get("expires_in", 1499))
        return self._token


def search(token: Token, params: dict, retries: int = 4) -> tuple[list, int, int]:
    """Returns (results, range_start, total). total is parsed from Content-Range
    ('offres a-b/total'); 200 = full result set, 206 = partial."""
    url = f"{SEARCH_URL}?{urlencode(params)}"
    backoff = 2
    for attempt in range(retries):
        req = Request(
            url, headers={"Authorization": f"Bearer {token.get()}", "Accept": "application/json"}
        )
        try:
            with urlopen(req, timeout=60) as resp:
                # 204 = no offers match this filter.
                if resp.status == 204:
                    return [], 0, 0
                payload = json.loads(resp.read().decode("utf-8"))
                cr = resp.headers.get("Content-Range", "")  # e.g. "offres 0-149/587780"
                total = (
                    int(cr.rsplit("/", 1)[-1]) if "/" in cr else len(payload.get("resultats", []))
                )
                return payload.get("resultats", []) or [], 0, total
        except HTTPError as e:
            if e.code == 204:
                return [], 0, 0
            # 400 with a bad range still means "out of results"; treat as empty.
            if e.code == 400 and "range" in url:
                return [], 0, 0
            if 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == retries - 1:
                raise
            sys.stderr.write(f"  {params} HTTP {e.code}; sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
        except (URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            sys.stderr.write(f"  {params} {e}; sleeping {backoff}s\n")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


def transform(offer: dict) -> dict[str, Any]:
    offer_id = str(offer.get("id") or "")
    company = (offer.get("entreprise") or {}).get("nom") or ""
    title = strip_tags(offer.get("intitule") or "")

    lieu = offer.get("lieuTravail") or {}
    # libelle looks like "75 - PARIS" or "Paris (75)"; keep the place, drop the bare
    # departement code so locations read like the other sources.
    libelle = (lieu.get("libelle") or "").strip()
    place = re.sub(r"^\d{2,3}\s*-\s*", "", libelle).strip()
    locations = [place] if place else []

    ct = offer.get("typeContratLibelle") or offer.get("typeContrat")
    duree = offer.get("dureeTravailLibelleConverti")  # "Temps plein" / "Temps partiel"
    employment_type = " / ".join([b for b in (ct, duree) if b]) or None

    # No structured remote flag; infer from the French term, mirroring fetch_adzuna.
    hay = f"{title} {offer.get('description') or ''}".lower()
    remote = (
        True if ("télétravail" in hay or "teletravail" in hay or "100% remote" in hay) else None
    )

    return {
        "id": f"francetravail:{offer_id}",
        "source_slug": slugify(company) or "france-travail",
        "title": title,
        # description is plain text; prep strips HTML from this field anyway.
        "description_html": offer.get("description") or "",
        "department": offer.get("secteurActiviteLibelle") or None,  # industry analog
        "employment_type": employment_type,
        "remote": remote,
        "locations": locations,
        "salary_min": None,  # salaire is free-text ("salaire.libelle"), not parseable min/max
        "salary_max": None,
        "salary_currency": None,
        "posted_at": offer.get("dateCreation") or None,  # ISO 8601
        "source": "francetravail",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--publiee-depuis",
        type=int,
        default=7,
        choices=[1, 3, 7, 14, 31],
        help="only offers published within the last N days (API allows 1/3/7/14/31)",
    )
    ap.add_argument(
        "--departements",
        default="",
        help="comma-separated departement codes to limit to (default: all of France)",
    )
    ap.add_argument("--rows-per-file", type=int, default=10_000)
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between calls (be polite)")
    args = ap.parse_args()

    cid = os.environ.get("FRANCETRAVAIL_CLIENT_ID")
    secret = os.environ.get("FRANCETRAVAIL_SECRET")
    if not cid or not secret:
        sys.exit("ERROR: set FRANCETRAVAIL_CLIENT_ID and FRANCETRAVAIL_SECRET env vars")

    deps = [d.strip() for d in args.departements.split(",") if d.strip()] or DEPARTEMENTS
    token = Token(cid, secret)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    buf: list[dict] = []
    file_idx = 0
    n_written = 0
    seen_ids: set[str] = set()

    def flush():
        nonlocal buf, file_idx, n_written
        if not buf:
            return
        table = pa.Table.from_pylist(buf)
        path = out / f"francetravail-{file_idx:04d}.parquet"
        pq.write_table(table, path)
        n_written += len(buf)
        print(f"  wrote {path.name}: {len(buf):,} rows (total {n_written:,})", flush=True)
        file_idx += 1
        buf = []

    for dep in deps:
        start = 0
        dep_rows = 0
        total = None
        while start < RANGE_CAP:
            end = start + PAGE - 1
            params = {
                "departement": dep,
                "publieeDepuis": args.publiee_depuis,
                "range": f"{start}-{end}",
            }
            results, _, total = search(token, params)
            time.sleep(args.sleep)
            if not results:
                break
            for offer in results:
                row = transform(offer)
                if not row["title"] or row["id"] in seen_ids:
                    continue
                seen_ids.add(row["id"])
                buf.append(row)
                dep_rows += 1
            if len(buf) >= args.rows_per_file:
                flush()
            start += PAGE
            if total is not None and start >= total:
                break
        if total and total >= RANGE_CAP:
            sys.stderr.write(
                f"  [dep {dep}] {total:,} offers exceeds the {RANGE_CAP} range cap; "
                f"narrow --publiee-depuis to capture the rest\n"
            )
        if dep_rows:
            print(
                f"[dep {dep}] collected {dep_rows:,} postings (total found {total:,})", flush=True
            )

    flush()
    print(
        f"done: {n_written:,} unique postings across {file_idx} parquet files -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
