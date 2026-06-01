"""Shared snippet passage logic — the SINGLE source of truth for how a job
description is cleaned and split into candidate passages, used by BOTH:

  * the offline index-time encoder (encode_snippet_vecs.py / push_docs.py), which
    embeds each passage with e5-small and packs the vectors into the stored
    `snippet_vecs` Solr field, and
  * the serving path (app.py), which re-derives the SAME passages from the stored
    description and pairs passage[i] with the stored vector[i].

Because both call `passages_for()` on the same raw description with the same
constants, vector[i] always lines up with passage[i]. If this segmentation ever
changes, the stored vectors go stale (misaligned) until the next full rebuild —
so app.py guards on a length match and falls back to live encoding on mismatch.
"""

import base64
import re

import numpy as np

SNIPPET_LEN = 240  # target passage length (chars); also the display window width
PASSAGES_PER_DOC = 8  # cap candidate passages per description (bounds encode cost + index size)
SNIPPET_PASSAGE_PREFIX = "passage: "  # must match the catalog's e5 "passage: " encoding
SNIPPET_VEC_DIM = 384  # e5-small-v2 dimension

_WS_RUN = re.compile(r"[ \t]+")
_NL_RUN = re.compile(r"\n{3,}")
_SNIP_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_text(s: str) -> str:
    """Decode HTML entities, collapse whitespace. Idempotent. (app.py's _clean_text
    delegates here so the offline encode and the live snippet see identical text.)"""
    if not s:
        return ""
    import html

    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    s = _WS_RUN.sub(" ", s)
    s = _NL_RUN.sub("\n\n", s)
    return s.strip()


def passages(text: str) -> list[str]:
    """Segment cleaned text into coherent ~SNIPPET_LEN passages: greedily merge
    sentences until the next would overflow SNIPPET_LEN (so a candidate is a whole
    thought, not a fragment); a lone oversized sentence is its own passage. Capped at
    PASSAGES_PER_DOC."""
    sents = [s.strip() for s in _SNIP_SENT.split(text) if s.strip()]
    out: list[str] = []
    cur = ""
    for s in sents:
        cand = (cur + " " + s) if cur else s
        if cur and len(cand) > SNIPPET_LEN:
            out.append(cur)
            cur = s
        else:
            cur = cand
        if len(out) >= PASSAGES_PER_DOC:
            return out[:PASSAGES_PER_DOC]
    if cur and len(out) < PASSAGES_PER_DOC:
        out.append(cur)
    return out


def passages_for(raw_description: str) -> list[str]:
    """Clean + segment a raw description into candidate passages. The one call both
    the offline encoder and the serving path use, so passage order/count match."""
    return passages(clean_text(raw_description))


def pack_vecs(arr) -> str:
    """Pack a (n, dim) float array into a base64 string of little-endian fp16 bytes —
    the wire form for the Solr `snippet_vecs` BinaryField (stored as raw bytes on disk,
    not the inflated base64)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f2"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def unpack_vecs(b64: str, dim: int = SNIPPET_VEC_DIM) -> np.ndarray:
    """Inverse of pack_vecs: base64 fp16 bytes -> (n, dim) float32 array (cast up for
    fast dot products). Returns shape (0, dim) for empty/blank input."""
    if not b64:
        return np.empty((0, dim), dtype=np.float32)
    buf = base64.b64decode(b64)
    a = np.frombuffer(buf, dtype="<f2")
    if a.size % dim != 0:
        raise ValueError(f"snippet_vecs byte length {a.size} not a multiple of dim {dim}")
    return a.reshape(-1, dim).astype(np.float32)
