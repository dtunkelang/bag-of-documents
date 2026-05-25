#!/usr/bin/env python3
"""Solr-backed movies search demo.

BM25 over title/lead/plot/cast/directors/genres with edismax field boosts.
Facets: genres, decade, type, has_bag. No semantic lane yet (open-weight
encoder will go into the bge_vec slot in a later phase).
"""

import html
import os
from urllib.parse import urlencode

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

SOLR = os.environ.get("SOLR", "http://localhost:8984")
CORE = "movies"

QF = "title^6 original_title^4 cast_names^3 director_names^3 genres^2 lead^1 plot^0.5"
PF = "title^10 cast_names^4 director_names^4"

FACET_FIELDS = ["genres", "decade", "type", "has_bag"]
FACET_LIMIT = 8

app = FastAPI()


def solr_search(
    q: str,
    rows: int = 20,
    genres: list[str] | None = None,
    decade: str | None = None,
    typ: str | None = None,
    has_bag: bool | None = None,
) -> dict:
    if not q or not q.strip():
        return {"hits": 0, "docs": [], "facets": {}}
    params: list[tuple[str, str]] = [
        ("q", q),
        ("defType", "edismax"),
        ("qf", QF),
        ("pf", PF),
        ("rows", str(rows)),
        ("wt", "json"),
        (
            "fl",
            "id,title_display,year,type,genres,rating,votes,director_names,"
            "cast_names,lead,has_lead,has_bag",
        ),
        ("facet", "true"),
        ("facet.limit", str(FACET_LIMIT)),
        ("facet.mincount", "1"),
        # Popularity nudge: log-scaled votes as a multiplicative boost so BM25
        # still dominates relevance; adds at most ~log10(2.5M) ≈ 6.4 multiplier
        # for the most-voted titles, vanishes for the long tail.
        ("boost", "log(sum(def(votes,0),10))"),
    ]
    for f in FACET_FIELDS:
        params.append(("facet.field", f))
    if genres:
        for g in genres:
            params.append(("fq", f'genres:"{g}"'))
    if decade:
        params.append(("fq", f'decade:"{decade}"'))
    if typ:
        params.append(("fq", f'type:"{typ}"'))
    if has_bag is not None:
        params.append(("fq", f"has_bag:{str(has_bag).lower()}"))

    r = requests.get(f"{SOLR}/solr/{CORE}/select", params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    resp = payload.get("response", {})
    facets: dict[str, list[tuple[str, int]]] = {}
    for f, vs in payload.get("facet_counts", {}).get("facet_fields", {}).items():
        facets[f] = list(zip(vs[::2], vs[1::2]))
    return {"hits": resp.get("numFound", 0), "docs": resp.get("docs", []), "facets": facets}


@app.get("/api/search")
def api_search(
    q: str = Query(""),
    rows: int = Query(20, ge=1, le=100),
    genres: list[str] = Query(default=[]),
    decade: str = Query(""),
    typ: str = Query("", alias="type"),
    has_bag: bool | None = Query(None),
) -> JSONResponse:
    out = solr_search(
        q,
        rows=rows,
        genres=genres or None,
        decade=decade or None,
        typ=typ or None,
        has_bag=has_bag,
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
      <div class="title">{title} <span class="id">{html.escape(d.get("id", ""))}</span></div>
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
    return "/?" + urlencode(params) if params else "/"


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
            href = _facet_url(q, genres, decade, typ, has_bag, toggle=(f, sv))
            cls = " class='sel'" if selected else ""
            mark = "✓ " if selected else ""
            items_html.append(
                f"<li><a href='{html.escape(href)}'{cls}>"
                f"{mark}{html.escape(sv)} <span class='n'>{n:,}</span></a></li>"
            )
        clear_html = ""
        if active[f]:
            clear_href = _facet_url(q, genres, decade, typ, has_bag, clear=f)
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
):
    has_bag_filter: bool | None
    if has_bag == "true":
        has_bag_filter = True
    elif has_bag == "false":
        has_bag_filter = False
    else:
        has_bag_filter = None
    out = (
        solr_search(
            q,
            rows=20,
            genres=genres or None,
            decade=decade or None,
            typ=typ or None,
            has_bag=has_bag_filter,
        )
        if q
        else {"hits": 0, "docs": [], "facets": {}}
    )
    docs_html = "\n".join(_render_doc(d) for d in out["docs"])
    facets_html = _render_facets(out["facets"], q, genres, decade, typ, has_bag)
    summary = f"<p class='summary'>{out['hits']:,} hits</p>" if q else ""
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
</style>
</head>
<body>
<header>
  <form action="/" method="get">
    <input type="text" name="q" value="{html.escape(q)}" placeholder="search movies & TV (title, cast, plot)..." autofocus>
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
