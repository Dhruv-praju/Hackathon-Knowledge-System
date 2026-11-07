"""
Keyword-search solution using BM25 ranking.

Keeps every snapshot, so `as_of` queries are answered from the snapshot that was current on
that date. Paragraphs are ranked with BM25 over lightly stemmed, stop-word-filtered tokens.
Conflict detection is not implemented.
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

TOKEN = re.compile(r"[a-z0-9$%]+(?:\.[0-9]+)?%?")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or the to what when "
    "where which who will with you your".split()
)
K1, B = 1.5, 0.75
MIN_COVERAGE = 0.6
MIN_SCORE = 2.0


def stem(word):
    """Strip common English suffixes so 'costs' and 'cost' match."""
    for suffix in ("ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    """Lowercase, split into tokens, drop stop-words, and stem the rest."""
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class _Index:
    """BM25 index over the paragraphs of one snapshot."""

    def __init__(self, snapshot_id, pages):
        """Split each page into paragraphs and precompute term statistics."""
        self.snapshot_id = snapshot_id
        self.chunks = []  # (url, paragraph, term counts)
        for url, md in pages.items():
            for para in md.split("\n\n"):
                para = para.strip()
                tokens = tokenize(para)
                if tokens:
                    self.chunks.append((url, para, Counter(tokens)))
        n = len(self.chunks)
        self.avg_len = sum(sum(c.values()) for _, _, c in self.chunks) / n if n else 0
        df = Counter(t for _, _, c in self.chunks for t in c)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def best(self, query_tokens):
        """Return (url, paragraph, score, coverage) for the top paragraph, or None if nothing matches.

        `coverage` is the idf-weighted fraction of query terms found in that paragraph. Terms
        absent from the whole snapshot get the maximum idf, so they pull coverage down.
        """
        terms = set(query_tokens)
        unknown_idf = math.log(1 + (len(self.chunks) + 0.5) / 0.5)
        weights = {t: self.idf.get(t, unknown_idf) for t in terms}
        total_weight = sum(weights.values())
        best, best_score = None, 0.0
        for url, para, counts in self.chunks:
            length = sum(counts.values())
            score = matched = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    matched += weights[t]
                    score += self.idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self.avg_len))
            if score > best_score:
                best, best_score = (url, para, score, matched / total_weight), score
        return best


class KeywordKnowledge:
    """Knowledge base answering questions by BM25 keyword search over website snapshots."""

    def __init__(self):
        """Initialize empty snapshot history: a list of (taken_on, snapshot_id, {url: markdown})."""
        self.snapshots = []
        self._indexes = {}

    def ingest(self, snapshot: Snapshot) -> None:
        """Store a full snapshot of the site; called once per snapshot, oldest first."""
        pages = {p.url: p.markdown for p in snapshot.pages}
        self.snapshots.append((snapshot.taken_on, snapshot.snapshot_id, pages))

    def _index_for(self, as_of):
        """Return the index for the latest snapshot taken on or before `as_of` (latest if None)."""
        if not self.snapshots:
            return None
        chosen = self.snapshots[-1]
        if as_of:
            eligible = [s for s in self.snapshots if s[0] <= as_of]
            if not eligible:
                return None
            chosen = eligible[-1]
        _, sid, pages = chosen
        if sid not in self._indexes:
            self._indexes[sid] = _Index(sid, pages)
        return self._indexes[sid]

    def answer(self, question, as_of=None):
        """Answer with the best BM25-matching paragraph from the snapshot current at `as_of`.

        Args:
            question: Natural-language question to answer.
            as_of: Optional ISO date; answer as the website stood on that date.

        Returns:
            An Answer with the best paragraph (up to 600 chars) and its source, or a
            non-confident "I don't know." if the best match scores below MIN_SCORE or
            covers less than MIN_COVERAGE of the question's (idf-weighted) keywords.
        """
        index = self._index_for(as_of)
        hit = index.best(tokenize(question)) if index else None
        if not hit:
            return Answer(text="I don't know.", confident=False)
        url, para, score, coverage = hit
        if score < MIN_SCORE or coverage < MIN_COVERAGE:
            return Answer(text="I don't know.", confident=False)
        return Answer(
            text=para[:600],
            sources=[Source(url=url, snapshot_id=index.snapshot_id, quote=para[:200])],
        )

    def conflicts(self):
        """Return detected conflicts between pages (none; not implemented)."""
        return []


def build():
    """Create and return the knowledge-base instance used by the harness."""
    return KeywordKnowledge()
