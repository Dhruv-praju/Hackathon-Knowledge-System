"""
RAG with an LLM judge at answer time (Approach 4).

Ingest: parse tables into rows (free, no LLM) and keep paragraphs per snapshot. Nothing else.

Answer:
    1. Pick the snapshot current at `as_of` (newest if None).
    2. Retrieve: exact table rows (tier/term/year aware) + the top BM25 pages and their best paragraphs.
    3. Ask an LLM to read ONLY that context and decide the answer, following authority rules in the
       system prompt (newest page wins, official rate sheet beats blogs/press, respect qualifiers,
       expired promos, say "I don't know" when the site doesn't say).
    4. Answers are cached by (snapshot, question) so asking twice or
       re-running gives the same answer. Any LLM failure falls back to the table-only answer.

Setup:  pip install anthropic   and   export ANTHROPIC_API_KEY=...
Env:    RAG_MODEL (default claude-haiku-5-5), RAG_TIMEOUT_S (default 12)
"""
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Snapshot, Source  # noqa: E402
import table_facts as T  # noqa: E402

try:
    import anthropic
except ImportError:
    anthropic = None

MODEL = os.environ.get("RAG_MODEL", "claude-haiku-5-5")
TIMEOUT_S = float(os.environ.get("RAG_TIMEOUT_S", "12"))
PROMPT_VERSION = "v1"
CACHE_FILE = Path(os.environ.get("RAG_CACHE", HERE / "cache" / "answers.json"))
TOP_PAGES = 8
CHUNKS_PER_PAGE = 2
CHUNK_CHARS = 1200
CONTEXT_CHARS = 20000

SYSTEM = """You are a careful credit-union support expert answering a member's question from website excerpts.
Each excerpt shows its URL and the date it was crawled. Use ONLY the excerpts. Rules:
1. CURRENT: when sources disagree, prefer the newest crawl date, and an official rate sheet (/rates, /business-rates, a product page's rate table) over blog posts, newsletters or press releases. Old numbers on old pages are stale: never quote them.
2. EXPIRED: ignore promotions, events and offers that the text says have ended or whose end date has passed.
3. QUALIFIERS: respect the exact product (personal vs business, Share Savings vs Bright Savings), term, balance tier, loan amount, model year, purchase vs refinance, and rate vs APY vs APR. Different tiers are different facts, not conflicts. If the member gives an amount or term, give the matching tier; otherwise give the headline tier and mention the others briefly.
4. LISTS: for "what are all/the rates/cards/options" questions, list every item found in the excerpts.
5. HONEST: if the excerpts do not say, answer that the website does not state it and suggest contacting Dupaco; set confident=false. Never guess or invent a number, date, address or URL.
6. FORM: 1-4 short sentences (or a short list). Lead with the answer, include the exact figures as written (keep %, $, dates), no hedging, no preamble.
7. If you noticed two sources that really disagree, say which you used and why in conflict_note.
Return your answer with the answer tool."""

TOOL = {
    "name": "answer",
    "description": "Give the final answer to the member.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confident": {"type": "boolean", "description": "false if the excerpts do not contain the answer"},
            "source_urls": {"type": "array", "items": {"type": "string"}, "description": "URLs of excerpts the answer relies on"},
            "conflict_note": {"type": ["string", "null"]},
        },
        "required": ["answer", "confident", "source_urls"],
    },
}


WINDOW_CHARS = 900
LOW_VALUE = ("/calculators", "/learn/", "/business/resources", "/resources/", "/blog", "/tag/", "/author/")


class WindowIndex:
    """BM25 over sliding windows of consecutive paragraphs, one index per snapshot."""

    def __init__(self, snapshot):
        from collections import Counter
        import math

        self.windows = []  # (url, text, Counter, low_value)
        seen = set()
        for page in snapshot.pages:
            paras = [p.strip() for p in page.markdown.split("\n\n") if p.strip()]
            url_tokens = T.tokenize(page.url.split("dupaco.com", 1)[-1].replace("/", " ").replace("-", " ") + " " + page.title)
            low = any(m in page.url for m in LOW_VALUE)
            i = 0
            while i < len(paras):
                j, size = i, 0
                while j < len(paras) and (size == 0 or size + len(paras[j]) <= WINDOW_CHARS):
                    size += len(paras[j])
                    j += 1
                text = "\n".join(paras[i:j])
                key = text.strip().lower()
                if key not in seen and not text.lstrip().startswith("|"):
                    seen.add(key)
                    toks = T.tokenize(text) + url_tokens
                    if toks:
                        self.windows.append((page.url, text, Counter(toks), low))
                i = max(j - 1, i + 1)  # overlap by one paragraph
        df = Counter(t for _, _, c, _ in self.windows for t in c)
        n = max(len(self.windows), 1)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg_len = sum(sum(c.values()) for _, _, c, _ in self.windows) / n

    def search(self, question, k=40):
        """Top-k windows: list of (score, url, text)."""
        q = list(dict.fromkeys(T.tokenize(question)))
        out = []
        for url, text, counts, low in self.windows:
            length = sum(counts.values())
            score = 0.0
            for t in q:
                tf = counts.get(t, 0)
                if tf:
                    score += self.idf.get(t, 0.0) * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / self.avg_len))
            if score > 0:
                out.append((score * (0.6 if low else 1.0), url, text))
        out.sort(key=lambda x: -x[0])
        return out[:k]


class RagJudge:
    """KnowledgeSystem: table rows + BM25 pages -> LLM judge, with a table-only fallback."""

    def __init__(self):
        self.tables = T.FactStore()
        self.fetched = []  # per snapshot: {url: fetched_at}
        self.windows = []  # per snapshot: WindowIndex
        self.client = (
            anthropic.Anthropic(max_retries=1, timeout=TIMEOUT_S) if anthropic and os.environ.get("ANTHROPIC_API_KEY") else None
        )
        self.lock = threading.Lock()
        self.cache = {}
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text())
            except ValueError:
                self.cache = {}

    def ingest(self, snapshot: Snapshot) -> None:
        """Parse tables and index paragraphs for the snapshot; no LLM calls here."""
        self.tables.ingest(snapshot)
        self.fetched.append({p.url: p.fetched_at for p in snapshot.pages})
        self.windows.append(WindowIndex(snapshot))

    def _pick(self, as_of):
        """Index of the snapshot current at `as_of` (newest when None), or None."""
        stores = self.tables.stores
        if not stores:
            return None
        if not as_of:
            return len(stores) - 1
        ok = [i for i, s in enumerate(stores) if s.taken_on <= as_of]
        return ok[-1] if ok else None

    def _context(self, i, question):
        """Build the excerpt block the judge reads, and the set of URLs it contains."""
        store, dates = self.tables.stores[i], self.fetched[i]
        parts, urls = [], []
        rows, _ = store.lookup(question)
        if rows:
            parts.append(
                f"[RATE TABLE ROWS | official rate sheet | crawled {dates.get(rows[0].url, store.taken_on)[:10]} | {rows[0].url}]\n"
                + "\n".join(T.render(r) for r in rows)
            )
            urls.append(rows[0].url)
        by_page = {}
        for score, url, text in self.windows[i].search(question):
            by_page.setdefault(url, []).append((score, text))
        ranked = sorted(by_page.items(), key=lambda kv: -max(s for s, _ in kv[1]))
        for url, chunks in ranked[:TOP_PAGES]:
            chunks.sort(key=lambda c: -c[0])
            body = "\n---\n".join(t[:CHUNK_CHARS] for _, t in chunks[:CHUNKS_PER_PAGE])
            parts.append(f"[EXCERPT | crawled {dates.get(url, store.taken_on)[:10]} | {url}]\n{body}")
            urls.append(url)
        return "\n\n".join(parts)[:CONTEXT_CHARS], urls

    def _key(self, store, question):
        return hashlib.sha256(f"{PROMPT_VERSION}|{MODEL}|{store.snapshot_id}|{question.strip().lower()}".encode()).hexdigest()[:32]

    def _save(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(self.cache))
        except OSError:
            pass

    def answer(self, question, as_of=None):
        """Retrieve, let the LLM judge, cache. Falls back to the table-only answer on any failure."""
        i = self._pick(as_of)
        if i is None:
            return Answer(text="I don't know.", confident=False)
        store = self.tables.stores[i]
        if self.client is None:
            return self.tables.answer(question, as_of)
        key = self._key(store, question)
        with self.lock:
            hit = self.cache.get(key)
        if hit is None:
            context, urls = self._context(i, question)
            if not context:
                return Answer(text="I don't know.", confident=False)
            try:
                resp = self.client.messages.create(
                    model=MODEL,
                    max_tokens=500,
                    system=SYSTEM,
                    tools=[TOOL],
                    tool_choice={"type": "tool", "name": "answer"},
                    messages=[{"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"}],
                )
                out = next(b.input for b in resp.content if b.type == "tool_use")
            except Exception as exc:  # timeout / quota / network: degrade, don't crash
                print(f"  judge failed ({type(exc).__name__}): {exc}", file=sys.stderr)
                return self.tables.answer(question, as_of)
            allowed = set(urls)
            cited = [u for u in out.get("source_urls", []) if u in allowed] or urls[:1]
            hit = {
                "text": out["answer"],
                "confident": bool(out.get("confident", True)),
                "urls": cited,
                "note": out.get("conflict_note"),
            }
            with self.lock:
                self.cache[key] = hit
                self._save()
        return Answer(
            text=hit["text"],
            sources=[Source(url=u, snapshot_id=store.snapshot_id) for u in hit["urls"]],
            confident=hit["confident"],
            conflict_note=hit.get("note"),
        )

    def conflicts(self):
        """Table-row conflicts on the newest snapshot (official rate sheet wins)."""
        return self.tables.conflicts()


def build():
    """Create the knowledge system used by the harness."""
    return RagJudge()
