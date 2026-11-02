"""Knowledge base: markdown ingestion, section-aware chunking and hybrid retrieval.

Retrieval is BM25 (``rank_bm25``) by default, which needs no API key and works well
on short, keyword-dense support content. If an embeddings provider is configured,
dense similarity is fused with BM25 using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from had has have how i if in
    into is it its me my no not of on or our so that the their them then there these
    they this to too was we were what when where which who why will with would you your
    yours am been being did any all also just than very get got want need say says here
    hi hello please thanks thank help long take""".split()
)

# Irregular forms the suffix stemmer cannot handle.
IRREGULAR = {"broken": "break", "broke": "break", "wore": "wear", "worn": "wear", "bought": "buy",
             "paid": "pay", "sent": "send", "lost": "lose", "torn": "tear", "tore": "tear"}

# Domain synonyms used for query expansion. In a client project this lives in config and is
# grown from real search logs ("customers say X, our docs say Y").
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "boot": ["footwear"], "shoe": ["footwear"], "sneaker": ["footwear"], "runner": ["footwear"],
    "break": ["defect", "fail"], "tear": ["defect", "repair"], "leak": ["defect", "waterproof"],
    "money": ["refund"], "back": ["refund"], "late": ["delay"], "stolen": ["lose"],
    "coupon": ["promo"], "discount": ["promo"], "point": ["reward"], "fit": ["size"],
    "cost": ["price"], "match": ["adjustment"], "fee": ["price"], "wash": ["care"], "clean": ["care", "wash"],
}

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc-stem>#<section-slug>", used as the citation key
    doc: str  # file name, e.g. "shipping.md"
    title: str  # document title (H1)
    section: str  # section heading (H2) or "Overview"
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # normalised 0..1 relevance (1 = best match in the corpus for this query)
    raw_score: float  # absolute BM25 score
    coverage: float  # share of query terms (or their synonyms) found in the chunk, 0..1


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _stem(token: str) -> str:
    if token in IRREGULAR:
        return IRREGULAR[token]
    # Deliberately tiny stemmer: enough to match "returns"/"return", "shipping"/"ship".
    for suffix in ("ation", "ing", "es", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    else:
        if len(token) >= 7 and token.endswith("y"):  # delivery -> deliver, warranty -> warrant
            token = token[:-1]
    if len(token) > 3 and token[-1] == token[-2]:  # shipp -> ship
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def chunk_markdown(doc_name: str, markdown: str, max_chars: int = 1200) -> list[Chunk]:
    """Split a markdown document on H2 headings; long sections are split by paragraph."""
    stem = Path(doc_name).stem
    title_match = re.search(r"^#\s+(.+)$", markdown, flags=re.M)
    title = title_match.group(1).strip() if title_match else stem.replace("-", " ").title()

    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for line in markdown.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        parts = _split_long(body, max_chars)
        for i, part in enumerate(parts):
            slug = slugify(heading) + (f"-{i + 1}" if len(parts) > 1 else "")
            chunks.append(Chunk(f"{stem}#{slug}", doc_name, title, heading, part))
    return chunks


def _split_long(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", body):
        if current and len(current) + len(para) > max_chars:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], embed_fn: EmbedFn | None = None,
                 synonyms: dict[str, list[str]] | None = None):
        if not chunks:
            raise ValueError("knowledge base is empty")
        self.chunks = chunks
        self.synonyms = DEFAULT_SYNONYMS if synonyms is None else synonyms
        # Title and section are indexed with the body: they are strong relevance signals.
        corpus = [tokenize(f"{c.title} {c.section} {c.section} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._vocab = [set(doc) for doc in corpus]
        self._embed_fn = embed_fn
        self._vectors: list[list[float]] | None = None
        if embed_fn is not None:
            try:
                self._vectors = embed_fn([f"{c.title} - {c.section}\n{c.text}" for c in chunks])
            except Exception:  # embeddings are an optional boost, never a hard dependency
                log.exception("embedding the knowledge base failed; falling back to BM25 only")
                self._embed_fn = None

    @classmethod
    def from_directory(cls, path: Path, embed_fn: EmbedFn | None = None) -> "KnowledgeBase":
        chunks: list[Chunk] = []
        for file in sorted(Path(path).glob("**/*.md")):
            chunks.extend(chunk_markdown(file.name, file.read_text(encoding="utf-8")))
        return cls(chunks, embed_fn)

    @property
    def mode(self) -> str:
        return "hybrid (bm25 + embeddings)" if self._vectors is not None else "bm25"

    @property
    def documents(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        tokens = terms + [syn for t in terms for syn in self.synonyms.get(t, [])]
        bm25 = [float(x) for x in self._bm25.get_scores(tokens)]
        ranked = sorted(range(len(self.chunks)), key=lambda i: bm25[i], reverse=True)
        ranked = [i for i in ranked if bm25[i] > 0]

        if self._vectors is not None and self._embed_fn is not None:
            try:
                qv = self._embed_fn([query])[0]
                dense = [_cosine(qv, v) for v in self._vectors]
                dense_ranked = sorted(range(len(self.chunks)), key=lambda i: dense[i], reverse=True)
                ranked = _rrf([ranked, dense_ranked[: max(top_k * 3, 10)]])
            except Exception:
                log.exception("query embedding failed; using BM25 ranking")

        if not ranked:
            return []
        best = max(bm25[ranked[0]], 1e-9)
        return [
            SearchResult(self.chunks[i], round(min(bm25[i] / best, 1.0), 3), round(bm25[i], 3),
                         round(self._coverage(terms, i), 3))
            for i in ranked[:top_k]
        ]

    def _coverage(self, terms: list[str], idx: int) -> float:
        """IDF-weighted share of query terms present in the chunk (synonyms count as a match).

        Weighting by IDF means a rare term the corpus has never seen ("Brazil", "crypto")
        drags confidence down, while common filler terms barely matter.
        """
        vocab = self._vocab[idx]
        max_idf = max(self._bm25.idf.values())
        weights = {t: max(self._bm25.idf.get(t, max_idf), 0.1) for t in terms}
        hit = sum(w for t, w in weights.items()
                  if t in vocab or any(s in vocab for s in self.synonyms.get(t, [])))
        return hit / sum(weights.values())


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def openai_embedder(model: str, base_url: str | None = None) -> EmbedFn:
    from openai import OpenAI

    client = OpenAI(base_url=base_url or None)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]

    return embed
