#!/usr/bin/env python3
"""Solr-backed movies search demo.

Two retrieval lanes:
  - BM25 (edismax) over title/lead/plot/cast/directors/genres
  - KNN over bge_vec (bge-small-en-v1.5, 384-dim cosine) for the ~200K titles
    with Wikipedia leads
Fused via Reciprocal Rank Fusion (RRF, k=60). Facets are computed from the
BM25 lane and remain clickable; KNN expands the result set with semantically
similar titles that BM25 misses.
"""

import html
import os
from functools import lru_cache
from urllib.parse import urlencode

import numpy as np
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

SOLR = os.environ.get("SOLR", "http://localhost:8984")
CORE = "movies"

QF = "title^6 original_title^4 cast_names^3 director_names^3 genres^2 lead^1 plot^0.5"
PF = "title^10 cast_names^4 director_names^4"

FACET_FIELDS = ["genres", "decade", "type", "has_bag"]
FACET_LIMIT = 8

BGE_MODEL = os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5")
KNN_TOPK = int(os.environ.get("KNN_TOPK", "100"))
BM25_TOPK = int(os.environ.get("BM25_TOPK", "100"))
RRF_K = int(os.environ.get("RRF_K", "60"))
W_BM25 = float(os.environ.get("W_BM25", "1.0"))
W_KNN = float(os.environ.get("W_KNN", "2.0"))
# Soft lane gating: when BM25 returns >= GATE_BM25_NOISE hits, the query is
# probably a loose lexical match and KNN should be weighted further. The KNN
# weight is multiplied by GATE_BOOST when BM25 hits exceed the threshold.
GATE_BM25_NOISE = int(os.environ.get("GATE_BM25_NOISE", "5000"))
GATE_BOOST = float(os.environ.get("GATE_BOOST", "1.5"))

FL_FIELDS = (
    "id,title_display,year,type,genres,rating,votes,director_names,cast_names,lead,has_lead,has_bag"
)

app = FastAPI()

_encoder = None


def _get_encoder():
    """Lazy bge-small load; returns None if sentence-transformers/torch unusable."""
    global _encoder
    if _encoder is not None:
        return _encoder if _encoder is not False else None
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _encoder = SentenceTransformer(BGE_MODEL, device=device)
        _encoder.max_seq_length = 384
        print(f"loaded {BGE_MODEL} on {device}")
        return _encoder
    except Exception as e:
        print(f"encoder unavailable, KNN lane disabled: {e}")
        _encoder = False
        return None


@lru_cache(maxsize=512)
def _encode_query(q: str) -> tuple[float, ...] | None:
    enc = _get_encoder()
    if enc is None:
        return None
    v = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    return tuple(float(x) for x in v.astype(np.float32))


def _filter_params(
    genres: list[str] | None,
    decade: str | None,
    typ: str | None,
    has_bag: bool | None,
) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    if genres:
        for g in genres:
            params.append(("fq", f'genres:"{g}"'))
    if decade:
        params.append(("fq", f'decade:"{decade}"'))
    if typ:
        params.append(("fq", f'type:"{typ}"'))
    if has_bag is not None:
        params.append(("fq", f"has_bag:{str(has_bag).lower()}"))
    return params


def solr_bm25(
    q: str,
    rows: int,
    filters: list[tuple[str, str]],
    *,
    with_facets: bool,
) -> dict:
    """BM25 lane via edismax. Returns hits, ordered docs, and (optional) facets."""
    if not q or not q.strip():
        return {"hits": 0, "docs": [], "facets": {}}
    params: list[tuple[str, str]] = [
        ("q", q),
        ("defType", "edismax"),
        ("qf", QF),
        ("pf", PF),
        ("rows", str(rows)),
        ("wt", "json"),
        ("fl", FL_FIELDS),
        # Popularity nudge: log-scaled votes as a multiplicative boost so BM25
        # still dominates relevance; adds at most ~log10(2.5M) ≈ 6.4 multiplier
        # for the most-voted titles, vanishes for the long tail.
        ("boost", "log(sum(def(votes,0),10))"),
    ]
    if with_facets:
        params.append(("facet", "true"))
        params.append(("facet.limit", str(FACET_LIMIT)))
        params.append(("facet.mincount", "1"))
        for f in FACET_FIELDS:
            params.append(("facet.field", f))
    params.extend(filters)
    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    resp = payload.get("response", {})
    facets: dict[str, list[tuple[str, int]]] = {}
    for f, vs in payload.get("facet_counts", {}).get("facet_fields", {}).items():
        facets[f] = list(zip(vs[::2], vs[1::2]))
    return {"hits": resp.get("numFound", 0), "docs": resp.get("docs", []), "facets": facets}


def solr_knn(
    q: str,
    rows: int,
    filters: list[tuple[str, str]],
) -> dict:
    """KNN lane over bge_vec; returns hits + ordered docs (no facets)."""
    if not q or not q.strip():
        return {"hits": 0, "docs": []}
    vec = _encode_query(q)
    if vec is None:
        return {"hits": 0, "docs": []}
    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
    params: list[tuple[str, str]] = [
        ("q", f"{{!knn f=bge_vec topK={rows}}}{vec_str}"),
        ("rows", str(rows)),
        ("wt", "json"),
        ("fl", FL_FIELDS),
    ]
    params.extend(filters)
    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    resp = payload.get("response", {})
    return {"hits": resp.get("numFound", 0), "docs": resp.get("docs", [])}


def rrf_fuse(
    bm25_docs: list[dict],
    knn_docs: list[dict],
    rows: int,
    k: int = RRF_K,
    w_bm25: float = W_BM25,
    w_knn: float = W_KNN,
) -> list[dict]:
    """Weighted RRF over two ranked lists; tags each doc with its source lane(s)."""
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    by_id: dict[str, dict] = {}
    for rank, d in enumerate(bm25_docs):
        did = d.get("id")
        if not did:
            continue
        scores[did] = scores.get(did, 0.0) + w_bm25 / (k + rank + 1)
        sources.setdefault(did, set()).add("bm25")
        by_id.setdefault(did, d)
    for rank, d in enumerate(knn_docs):
        did = d.get("id")
        if not did:
            continue
        scores[did] = scores.get(did, 0.0) + w_knn / (k + rank + 1)
        sources.setdefault(did, set()).add("knn")
        by_id.setdefault(did, d)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])[:rows]
    out: list[dict] = []
    for did, _ in fused:
        d = dict(by_id[did])
        d["_sources"] = sorted(sources[did])
        out.append(d)
    return out


def search(
    q: str,
    rows: int = 20,
    genres: list[str] | None = None,
    decade: str | None = None,
    typ: str | None = None,
    has_bag: bool | None = None,
    lane: str = "hybrid",
) -> dict:
    """Top-level orchestrator. lane ∈ {hybrid, bm25, knn}."""
    if not q or not q.strip():
        return {"hits": 0, "docs": [], "facets": {}, "lane": lane}
    filters = _filter_params(genres, decade, typ, has_bag)
    if lane == "bm25":
        out = solr_bm25(q, rows=rows, filters=filters, with_facets=True)
        for d in out["docs"]:
            d["_sources"] = ["bm25"]
        out["lane"] = "bm25"
        return out
    if lane == "knn":
        bm = solr_bm25(q, rows=1, filters=filters, with_facets=True)
        out = solr_knn(q, rows=rows, filters=filters)
        for d in out["docs"]:
            d["_sources"] = ["knn"]
        out["facets"] = bm["facets"]
        out["lane"] = "knn"
        return out
    bm = solr_bm25(q, rows=BM25_TOPK, filters=filters, with_facets=True)
    kn = solr_knn(q, rows=KNN_TOPK, filters=filters)
    w_knn = W_KNN
    gated = bm["hits"] >= GATE_BM25_NOISE
    if gated:
        w_knn *= GATE_BOOST
    fused = rrf_fuse(bm["docs"], kn["docs"], rows=rows, w_knn=w_knn)
    return {
        "hits": bm["hits"],
        "knn_hits": kn["hits"],
        "docs": fused,
        "facets": bm["facets"],
        "lane": "hybrid",
        "fusion": {"w_bm25": W_BM25, "w_knn": w_knn, "gated": gated},
    }


# Back-compat shim for the existing /api/search caller.
def solr_search(
    q: str,
    rows: int = 20,
    genres: list[str] | None = None,
    decade: str | None = None,
    typ: str | None = None,
    has_bag: bool | None = None,
    lane: str = "hybrid",
) -> dict:
    return search(q, rows=rows, genres=genres, decade=decade, typ=typ, has_bag=has_bag, lane=lane)


@app.get("/api/search")
def api_search(
    q: str = Query(""),
    rows: int = Query(20, ge=1, le=100),
    genres: list[str] = Query(default=[]),
    decade: str = Query(""),
    typ: str = Query("", alias="type"),
    has_bag: bool | None = Query(None),
    lane: str = Query("hybrid"),
) -> JSONResponse:
    out = search(
        q,
        rows=rows,
        genres=genres or None,
        decade=decade or None,
        typ=typ or None,
        has_bag=has_bag,
        lane=lane,
    )
    return JSONResponse(out)


def _render_doc(d: dict) -> str:
    title = html.escape(d.get("title_display") or d.get("id", ""))
    year = d.get("year")
    rating = d.get("rating")
    votes = d.get("votes")
    typ = d.get("type") or ""
    genres = d.get("genres") or []
    directors = d.get("director_names") or []
    cast = (d.get("cast_names") or [])[:3]
    lead = d.get("lead") or ""
    snippet = (lead[:280] + "…") if len(lead) > 280 else lead
    badges = []
    if year:
        badges.append(f"{year}")
    if typ:
        badges.append(typ)
    if rating is not None:
        votes_s = f" / {votes:,} votes" if votes else ""
        badges.append(f"{rating}{votes_s}")
    if d.get("has_bag"):
        badges.append("bag")
    sources = d.get("_sources") or []
    src_html = ""
    if sources:
        src_class = "src-both" if len(sources) == 2 else f"src-{sources[0]}"
        src_html = f" <span class='src {src_class}'>{html.escape('+'.join(sources))}</span>"
    badge_html = " · ".join(html.escape(str(b)) for b in badges)
    genre_html = html.escape(", ".join(genres))
    people = []
    if directors:
        people.append("dir: " + html.escape(", ".join(directors[:2])))
    if cast:
        people.append("cast: " + html.escape(", ".join(cast)))
    people_html = " &nbsp;|&nbsp; ".join(people)
    return f"""
    <div class="doc">
      <div class="title">{title} <span class="id">{html.escape(d.get("id", ""))}</span>{src_html}</div>
      <div class="meta">{badge_html} &nbsp;|&nbsp; {genre_html}</div>
      <div class="people">{people_html}</div>
      <div class="snip">{html.escape(snippet)}</div>
    </div>
    """


def _facet_url(
    q: str,
    genres: list[str],
    decade: str,
    typ: str,
    has_bag: str,
    *,
    toggle: tuple[str, str] | None = None,
    clear: str | None = None,
    lane: str = "hybrid",
) -> str:
    """Build a `/?...` URL with one facet value toggled or cleared.

    Multi-select: genres (list). Single-select: decade, type, has_bag.
    """
    g = list(genres)
    d = decade
    t = typ
    h = has_bag
    if clear == "genres":
        g = []
    elif clear == "decade":
        d = ""
    elif clear == "type":
        t = ""
    elif clear == "has_bag":
        h = ""
    if toggle is not None:
        field, value = toggle
        if field == "genres":
            if value in g:
                g = [x for x in g if x != value]
            else:
                g = g + [value]
        elif field == "decade":
            d = "" if d == value else value
        elif field == "type":
            t = "" if t == value else value
        elif field == "has_bag":
            h = "" if h == value else value
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    for gv in g:
        params.append(("genres", gv))
    if d:
        params.append(("decade", d))
    if t:
        params.append(("type", t))
    if h:
        params.append(("has_bag", h))
    if lane and lane != "hybrid":
        params.append(("lane", lane))
    return "/?" + urlencode(params) if params else "/"


def _lane_links(
    q: str,
    genres: list[str],
    decade: str,
    typ: str,
    has_bag: str,
    current: str,
) -> str:
    parts = []
    for name in ("hybrid", "bm25", "knn"):
        params: list[tuple[str, str]] = []
        if q:
            params.append(("q", q))
        for gv in genres:
            params.append(("genres", gv))
        if decade:
            params.append(("decade", decade))
        if typ:
            params.append(("type", typ))
        if has_bag:
            params.append(("has_bag", has_bag))
        if name != "hybrid":
            params.append(("lane", name))
        href = "/?" + urlencode(params) if params else "/"
        cls = "lane-link sel" if name == current else "lane-link"
        parts.append(f"<a class='{cls}' href='{html.escape(href)}'>{name}</a>")
    return " ".join(parts)


def _is_selected(
    field: str,
    value: str,
    genres: list[str],
    decade: str,
    typ: str,
    has_bag: str,
) -> bool:
    if field == "genres":
        return value in genres
    if field == "decade":
        return value == decade
    if field == "type":
        return value == typ
    if field == "has_bag":
        return value == has_bag
    return False


def _render_facets(
    facets: dict[str, list[tuple[str, int]]],
    q: str,
    genres: list[str],
    decade: str,
    typ: str,
    has_bag: str,
    lane: str = "hybrid",
) -> str:
    blocks = []
    active = {
        "genres": bool(genres),
        "decade": bool(decade),
        "type": bool(typ),
        "has_bag": bool(has_bag),
    }
    for f in FACET_FIELDS:
        rows = facets.get(f) or []
        if not rows:
            continue
        items_html = []
        for v, n in rows:
            sv = str(v)
            if not sv:
                continue
            selected = _is_selected(f, sv, genres, decade, typ, has_bag)
            href = _facet_url(q, genres, decade, typ, has_bag, toggle=(f, sv), lane=lane)
            cls = " class='sel'" if selected else ""
            mark = "✓ " if selected else ""
            items_html.append(
                f"<li><a href='{html.escape(href)}'{cls}>"
                f"{mark}{html.escape(sv)} <span class='n'>{n:,}</span></a></li>"
            )
        clear_html = ""
        if active[f]:
            clear_href = _facet_url(q, genres, decade, typ, has_bag, clear=f, lane=lane)
            clear_html = f" <a class='clear' href='{html.escape(clear_href)}'>clear</a>"
        blocks.append(
            f"<div class='facet'><h4>{html.escape(f)}{clear_html}</h4>"
            f"<ul>{''.join(items_html)}</ul></div>"
        )
    return "\n".join(blocks)


@app.get("/", response_class=HTMLResponse)
def index(
    q: str = "",
    genres: list[str] = Query(default=[]),
    decade: str = "",
    typ: str = Query("", alias="type"),
    has_bag: str = "",
    lane: str = "hybrid",
):
    has_bag_filter: bool | None
    if has_bag == "true":
        has_bag_filter = True
    elif has_bag == "false":
        has_bag_filter = False
    else:
        has_bag_filter = None
    out = (
        search(
            q,
            rows=20,
            genres=genres or None,
            decade=decade or None,
            typ=typ or None,
            has_bag=has_bag_filter,
            lane=lane,
        )
        if q
        else {"hits": 0, "docs": [], "facets": {}, "lane": lane}
    )
    docs_html = "\n".join(_render_doc(d) for d in out["docs"])
    facets_html = _render_facets(out["facets"], q, genres, decade, typ, has_bag, lane=lane)
    if q:
        knn_n = out.get("knn_hits")
        lane_now = out.get("lane", lane)
        knn_bit = f" / knn {knn_n:,}" if knn_n is not None else ""
        summary = (
            f"<p class='summary'>{out['hits']:,} hits{knn_bit} "
            f"<span class='lane-tag'>lane: {html.escape(lane_now)}</span> "
            f"{_lane_links(q, genres, decade, typ, has_bag, lane_now)}</p>"
        )
    else:
        summary = ""
    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>BoD movies demo</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; color: #222; }}
  header {{ padding: 16px 24px; background: #111; color: #fafafa; }}
  header form {{ display: flex; gap: 8px; }}
  header input[type=text] {{ flex: 1; padding: 10px; font-size: 16px; border-radius: 4px; border: 0; }}
  header button {{ padding: 10px 18px; font-size: 15px; border-radius: 4px; border: 0; background: #2a7; color: #fff; cursor: pointer; }}
  main {{ display: flex; gap: 24px; padding: 16px 24px; }}
  .results {{ flex: 1; }}
  .summary {{ color: #666; margin: 4px 0 12px; }}
  .doc {{ padding: 12px 0; border-bottom: 1px solid #eee; }}
  .doc .title {{ font-size: 17px; font-weight: 600; }}
  .doc .id {{ color: #999; font-weight: normal; font-size: 12px; margin-left: 8px; }}
  .doc .meta {{ color: #555; font-size: 13px; margin: 4px 0; }}
  .doc .people {{ color: #555; font-size: 13px; margin: 2px 0; }}
  .doc .snip {{ color: #333; font-size: 13px; margin-top: 6px; line-height: 1.45; }}
  aside {{ width: 240px; }}
  .facet {{ margin-bottom: 18px; }}
  .facet h4 {{ margin: 0 0 6px; font-size: 13px; text-transform: uppercase; color: #888; }}
  .facet ul {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
  .facet li {{ padding: 2px 0; }}
  .facet li a {{ display: flex; justify-content: space-between; text-decoration: none; color: #2255aa; padding: 1px 4px; border-radius: 3px; }}
  .facet li a:hover {{ background: #f0f4ff; }}
  .facet li a.sel {{ background: #2255aa; color: #fff; font-weight: 600; }}
  .facet li a.sel .n {{ color: #cfd9f2; }}
  .facet .n {{ color: #888; }}
  .facet .clear {{ font-size: 11px; color: #c33; text-decoration: none; margin-left: 6px; text-transform: none; font-weight: normal; }}
  .facet .clear:hover {{ text-decoration: underline; }}
  .src {{ font-size: 10px; padding: 1px 6px; border-radius: 3px; vertical-align: middle; margin-left: 6px; font-weight: 500; }}
  .src-bm25 {{ background: #fdecc8; color: #7a4f00; }}
  .src-knn {{ background: #d4e6f7; color: #1f4a7a; }}
  .src-both {{ background: #d6f0d6; color: #1f5f1f; }}
  .lane-tag {{ font-size: 11px; color: #888; margin-left: 12px; }}
  .lane-link {{ font-size: 12px; color: #2255aa; text-decoration: none; margin-left: 4px; padding: 1px 6px; border: 1px solid #ddd; border-radius: 3px; }}
  .lane-link.sel {{ background: #2255aa; color: #fff; border-color: #2255aa; font-weight: 600; }}
  .lane-link:hover {{ background: #f0f4ff; }}
  .lane-link.sel:hover {{ background: #2255aa; }}
</style>
</head>
<body>
<header>
  <form action="/" method="get">
    <input type="text" name="q" value="{html.escape(q)}" placeholder="search movies & TV (title, cast, plot)..." autofocus>
    {("<input type='hidden' name='lane' value='" + html.escape(lane) + "'>") if lane and lane != "hybrid" else ""}
    <button type="submit">Search</button>
  </form>
</header>
<main>
  <div class="results">
    {summary}
    {docs_html}
  </div>
  <aside>{facets_html}</aside>
</main>
</body></html>"""
    return HTMLResponse(page)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app", host="127.0.0.1", port=int(os.environ.get("PORT", "7865")), log_level="info"
    )
