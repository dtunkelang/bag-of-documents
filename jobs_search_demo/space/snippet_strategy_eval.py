#!/usr/bin/env python3
"""Offline bake-off of snippet PASSAGE-SELECTION strategies.

Holds the rest of the snippet pipeline fixed (same segmentation via snippet_lib, same
e5-small vectors — stored snippet_vecs when present, else live-encoded, same window+
highlight at display) and varies ONLY which passage gets picked. Pulls the real top-K
results per query from the running local search, then scores every strategy two ways:

  * proxy metrics  — highlight coverage, lexical hits, e5 cosine, cosine regret vs the
                     semantic-best passage, agreement with today's behavior.
  * LLM judge      — gpt-4o-mini rates each *distinct* chosen passage 0-3 on "does this
                     snippet show why this job matches the query" (the perceived-relevance
                     question proxies can't answer). Deduped + cached, temp 0.

Read-only against Solr; never touches the index or the app.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snippet_lib import passages_for, unpack_vecs  # shared source of truth

APP = "http://localhost:7860"
SOLR, CORE = "http://localhost:8983", "jobs"
DENSE_MODEL = "intfloat/e5-small-v2"
TOPK = 8  # results per query to snippet

# --- lexical scoring, copied verbatim from app.py so offline == serve ---
_SNIPPET_STOP = {
    "the",
    "and",
    "for",
    "with",
    "you",
    "your",
    "our",
    "are",
    "job",
    "jobs",
    "role",
    "roles",
    "work",
    "will",
    "this",
    "that",
    "from",
    "have",
    "all",
    "who",
}
_SNIP_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*")


def snippet_terms(query):
    out = []
    for w in _SNIP_TOKEN.findall(query.lower()):
        if len(w) > 1 and w not in _SNIPPET_STOP and w not in out:
            out.append(w)
    return out


def term_hit(word_lc, term):
    return word_lc == term or (len(term) >= 4 and word_lc.startswith(term))


def distinct_hits(text, terms):
    words = {w.lower() for w in _SNIP_TOKEN.findall(text)}
    return sum(1 for t in terms if any(term_hit(w, t) for w in words))


# --- query set: lexical, acronym, role+constraint, paraphrase, broad ---
QUERIES = [
    "python developer",
    "kubernetes engineer",
    "react frontend",
    "salesforce administrator",
    "registered nurse night shift",
    "certified public accountant",
    "sales development rep",
    "remote senior data scientist",
    "part time warehouse associate",
    "entry level marketing",
    "help people manage their money",
    "teaching young children",
    "build mobile apps",
    "keep the office running smoothly",
    "write content for websites",
    "sales",
    "nursing",
]


def unit(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n == 0, 1, n)


# ---------- strategies: each takes (cos[n], hits[n]) -> chosen index ----------
def s_semantic(cos, hits):  # current behavior
    return int(np.argmax(cos))


def s_lexical(cos, hits):  # old fallback: most distinct hits, earliest wins
    if max(hits) == 0:
        return 0  # lead
    best, bi = -1, 0
    for i, h in enumerate(hits):
        if h > best:
            best, bi = h, i
    return bi


def s_gated(cos, hits):  # restrict semantic argmax to term-containing passages
    cand = [i for i, h in enumerate(hits) if h > 0]
    if not cand:
        return int(np.argmax(cos))
    return max(cand, key=lambda i: cos[i])


def _weighted(lam):
    def f(cos, hits):
        return int(np.argmax(np.asarray(cos) + lam * np.asarray(hits, float)))

    return f


def _tiebreak(eps):
    def f(cos, hits):
        cap = max(cos) - eps
        cand = [i for i in range(len(cos)) if cos[i] >= cap]
        return max(cand, key=lambda i: (hits[i], cos[i]))

    return f


def s_rrf(cos, hits, k=60):
    n = len(cos)
    cos_rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -cos[i]))}
    hit_rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -hits[i]))}
    score = {i: 1 / (k + cos_rank[i]) + 1 / (k + hit_rank[i]) for i in range(n)}
    return max(range(n), key=lambda i: (score[i], cos[i]))


STRATEGIES = {
    "semantic(now)": s_semantic,
    "lexical(old)": s_lexical,
    "gated": s_gated,
    "weighted.05": _weighted(0.05),
    "weighted.10": _weighted(0.10),
    "weighted.20": _weighted(0.20),
    "tiebreak.03": _tiebreak(0.03),
    "rrf60": s_rrf,
}


def fetch_topk(query):
    r = requests.get(f"{APP}/api/search", params={"q": query}, timeout=30)
    r.raise_for_status()
    rows = r.json()["results"][:TOPK]
    return [row["idx"] for row in rows if row.get("idx", -1) >= 0]


def fetch_docs(ids):
    clause = " OR ".join(f'id:"{i}"' for i in ids)
    r = requests.get(
        f"{SOLR}/solr/{CORE}/select",
        params={"q": clause, "rows": len(ids), "fl": "id,description,snippet_vecs"},
        timeout=30,
    )
    r.raise_for_status()
    return {int(d["id"]): d for d in r.json()["response"]["docs"]}


def main():
    print(f"loading {DENSE_MODEL} ...", flush=True)
    model = SentenceTransformer(DENSE_MODEL, device="cpu")

    def encode(texts, prefix):
        return unit(
            np.asarray(
                model.encode(
                    [prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False
                ),
                dtype=np.float32,
            )
        )

    # build per (query, doc) candidate passages + cos + hits
    cases = []  # dict: query, idx, passages, cos[], hits[]
    live_encoded = 0
    for q in QUERIES:
        try:
            ids = fetch_topk(q)
        except Exception as e:
            print(f"  search failed for {q!r}: {e}", flush=True)
            continue
        docs = fetch_docs(ids)
        qv = encode([q], "query: ")[0]
        for i in ids:
            d = docs.get(i)
            if not d:
                continue
            ps = passages_for(d.get("description") or "")
            if not ps:
                continue
            b64 = d.get("snippet_vecs") or ""
            pv = None
            try:
                v = unpack_vecs(b64)
                if v.shape[0] == len(ps):
                    pv = unit(v)
            except Exception:
                pv = None
            if pv is None:
                pv = encode(ps, "passage: ")
                live_encoded += 1
            cos = (pv @ qv).tolist()
            terms = snippet_terms(q)
            hits = [distinct_hits(p, terms) for p in ps]
            cases.append({"query": q, "idx": i, "passages": ps, "cos": cos, "hits": hits})
    print(
        f"built {len(cases)} (query,doc) cases over {len(QUERIES)} queries "
        f"({live_encoded} docs live-encoded, rest from stored snippet_vecs)",
        flush=True,
    )

    # run strategies -> chosen passage index per case
    for c in cases:
        c["pick"] = {name: fn(c["cos"], c["hits"]) for name, fn in STRATEGIES.items()}

    # ---- LLM judge over the DISTINCT chosen passages per case ----
    judge_cache = {}
    to_judge = []  # (query, passage)
    for c in cases:
        for pi in c["pick"].values():
            key = (c["query"], c["passages"][pi])
            if key not in judge_cache and key not in to_judge:
                to_judge.append(key)
    print(
        f"judging {len(to_judge)} distinct (query,passage) pairs with gpt-4o-mini ...", flush=True
    )

    from openai import OpenAI

    client = OpenAI()
    SYS = (
        "You rate job-search result SNIPPETS. Given a user query and one snippet drawn "
        "from a job description, rate how well the snippet helps the user see WHY this job "
        "is relevant to their query (its perceived relevance / explanatory value). "
        "0=irrelevant or generic boilerplate; 1=weakly on-topic; 2=clearly relevant; "
        "3=directly answers the query intent. Reply with ONLY the integer."
    )

    def judge(key):
        q, passage = key
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=2,
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": f"Query: {q}\nSnippet: {passage}"},
                ],
            )
            m = re.search(r"[0-3]", resp.choices[0].message.content or "")
            return key, (int(m.group()) if m else 0)
        except Exception:
            return key, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for key, score in ex.map(judge, to_judge):
            judge_cache[key] = score

    # ---- aggregate ----
    def hl_available(c):  # is there ANY passage with a query-term hit?
        return max(c["hits"]) > 0

    rows = []
    for name in STRATEGIES:
        cov = covden = hitsum = cossum = regret = agree = jsum = jn = 0
        n = len(cases)
        for c in cases:
            pi = c["pick"][name]
            h = c["hits"][pi]
            hitsum += h
            cossum += c["cos"][pi]
            regret += max(c["cos"]) - c["cos"][pi]
            agree += 1 if pi == c["pick"]["semantic(now)"] else 0
            if hl_available(c):
                covden += 1
                cov += 1 if h > 0 else 0
            j = judge_cache.get((c["query"], c["passages"][pi]))
            if j is not None:
                jsum += j
                jn += 1
        rows.append(
            {
                "strategy": name,
                "judge": jsum / jn if jn else float("nan"),
                "hl_cov": cov / covden if covden else float("nan"),
                "hits": hitsum / n,
                "cos": cossum / n,
                "regret": regret / n,
                "agree": agree / n,
            }
        )

    rows.sort(key=lambda r: -r["judge"])
    print(f"\n================ STRATEGY BAKE-OFF ({len(cases)} query-doc cases) ================")
    print(
        f"{'strategy':14} {'JUDGE':>6} {'hl_cov':>7} {'hits':>5} {'cos':>6} "
        f"{'regret':>7} {'agree':>6}"
    )
    print(f"{'':14} {'0-3':>6} {'%hili':>7} {'/snp':>5} {'e5':>6} {'vs.best':>7} {'w/now':>6}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['strategy']:14} {r['judge']:6.2f} {r['hl_cov'] * 100:6.0f}% "
            f"{r['hits']:5.2f} {r['cos']:6.3f} {r['regret']:7.3f} {r['agree'] * 100:5.0f}%"
        )

    # disagreement examples: where gated != semantic AND judge differs
    print("\n================ DISAGREEMENT EXAMPLES (semantic vs gated) ================")
    shown = 0
    for c in cases:
        sp, gp = c["pick"]["semantic(now)"], c["pick"]["gated"]
        if sp == gp:
            continue
        js = judge_cache.get((c["query"], c["passages"][sp]))
        jg = judge_cache.get((c["query"], c["passages"][gp]))
        if js is None or jg is None or js == jg:
            continue
        shown += 1
        if shown > 12:
            break
        print(f"\nQ: {c['query']!r}  (doc {c['idx']})")
        print(
            f"  semantic [judge {js}, cos {c['cos'][sp]:.3f}, hits {c['hits'][sp]}]: {c['passages'][sp][:200]}"
        )
        print(
            f"  gated    [judge {jg}, cos {c['cos'][gp]:.3f}, hits {c['hits'][gp]}]: {c['passages'][gp][:200]}"
        )

    out = {
        "cases": cases,
        "summary": rows,
        "judge_cache": {f"{k[0]} ||| {k[1]}": v for k, v in judge_cache.items()},
    }
    with open("snippet_strategy_eval_out.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote snippet_strategy_eval_out.json", flush=True)


if __name__ == "__main__":
    main()
