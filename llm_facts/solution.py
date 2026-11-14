"""
LLM-powered fact extraction at ingest time (Approach 3).

Ingest (untimed, <30 min)
    * Table rows are parsed deterministically by `table_facts` (Approach 2), which is free and exact.
    * Every other fact-bearing page (branches, hours, newsroom, product pages, ...) is read by an LLM
      that returns structured facts: entity, attribute, value, qualifiers, effective date, quote.
    * Results are cached on disk by page-content hash, so a page unchanged across snapshots is
      extracted once and re-runs are free. Without an API key, only the table facts are used.

Answer (fast, no LLM)
    1. table lookup (exact tier / year / term handling)      -> table_facts.SnapshotStore.lookup
    2. LLM-fact lookup (token recall over entity+attribute)  -> LLMFactIndex.lookup
    3. paragraph BM25 fallback, else an honest "I don't know".

Setup:  pip install anthropic   and   export ANTHROPIC_API_KEY=...
Env:    LLM_FACTS_MODEL (default claude-haiku-5-5), LLM_FACTS_WORKERS (default 8),
        LLM_FACTS_BUDGET_S (default 1500: stop calling the LLM after this many seconds of ingest).
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
import table_facts as T  # noqa: E402

try:
    import anthropic
except ImportError:
    anthropic = None

MODEL = os.environ.get("LLM_FACTS_MODEL", "claude-haiku-5-5")
WORKERS = int(os.environ.get("LLM_FACTS_WORKERS", "8"))
BUDGET_S = float(os.environ.get("LLM_FACTS_BUDGET_S", "1500"))
CACHE_DIR = Path(os.environ.get("LLM_FACTS_CACHE", HERE / "cache"))
MAX_PAGE_CHARS = 12000
PROMPT_VERSION = "v1"  # bump to invalidate the cache when the prompt changes

SKIP_URL = re.compile(
    r"/learn/|/calculators?|/business/resources|/business-resources|privacy-policy|/tag/|/blog|/author/|/page/\d|/search|/resources/"
)

SYSTEM = (
    "You extract member-relevant facts from one page of a credit union website. "
    "Return every concrete, checkable fact a member might ask about: rates, fees, limits, minimums, "
    "hours, addresses, phone numbers, routing numbers, branch counts, eligibility rules, program benefits, "
    "closures and other dated events, award counts, product features, and URLs to use for a task. "
    "Rules: one fact per item; copy values exactly as written (keep %, $, dates); put tier/term/branch/"
    "product distinctions in `qualifiers`; set `effective` to the date the fact applies from or an "
    "'as of' date if the page states one, else null; `quote` is a short verbatim excerpt (<=200 chars) "
    "that supports the fact. Do not invent facts. Skip navigation, marketing fluff and anything you "
    "cannot support with a quote. Mark facts that are explicitly expired, ended or superseded with "
    "status='expired'; otherwise status='current'. Skip rate tables: they are parsed separately."
)

TOOL = {
    "name": "record_facts",
    "description": "Record the structured facts found on the page.",
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "What the fact is about, e.g. 'Galena Branch', 'Hy-Vee branch', 'Dupaco'"},
                        "attribute": {"type": "string", "description": "e.g. 'address', 'lobby hours', 'closing date', 'routing number'"},
                        "value": {"type": "string"},
                        "qualifiers": {"type": "string", "description": "tier/term/day/branch qualifiers, or empty"},
                        "effective": {"type": ["string", "null"]},
                        "status": {"type": "string", "enum": ["current", "expired"]},
                        "quote": {"type": "string"},
                    },
                    "required": ["entity", "attribute", "value", "quote"],
                },
            }
        },
        "required": ["facts"],
    },
}


DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20\d\d)\b")
MONTHS = {m: i + 1 for i, m in enumerate("January February March April May June July August September October November December".split())}
QUESTION_NOISE = frozenset("many number count total become join".split())  # say nothing about WHICH fact
NORMALIZE = {"eligible": "eligib", "eligibility": "eligib"}
PERCENT = T.PERCENT


def page_date(page):
    """Publication date (YYYY-MM-DD) of an article page, else '' (undated pages never outrank dated ones)."""
    m = DATE.search(page.markdown[:1500])
    return f"{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}" if m else ""


def toks(text):
    """T.tokenize plus a light eligible/eligibility normalisation."""
    return [NORMALIZE.get(t, t) for t in T.tokenize(text)]


def page_hash(page):
    """Cache key: content + prompt version + model."""
    return hashlib.sha256(f"{PROMPT_VERSION}|{MODEL}|{page.url}|{page.markdown}".encode()).hexdigest()[:32]


def cache_path(key):
    """Path of the cached extraction for a key."""
    return CACHE_DIR / f"{key}.json"


def wants_llm(page):
    """True for pages worth an LLM call (prose-bearing, not blog/calculator/etc.)."""
    return not SKIP_URL.search(page.url) and len(page.markdown.strip()) > 200


class Extractor:
    """Calls the LLM once per unique page and caches the result."""

    def __init__(self):
        self.client = anthropic.Anthropic(max_retries=6) if anthropic and os.environ.get("ANTHROPIC_API_KEY") else None
        self.lock = threading.Lock()
        self.calls = self.cached = self.failed = 0

    @property
    def enabled(self):
        """True when an API client is available."""
        return self.client is not None

    def extract(self, page):
        """Return the list of fact dicts for a page (from cache, or via the LLM)."""
        key = page_hash(page)
        path = cache_path(key)
        if path.exists():
            with self.lock:
                self.cached += 1
            return json.loads(path.read_text())
        if not self.enabled:
            return []
        try:
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                tools=[TOOL],
                tool_choice={"type": "tool", "name": "record_facts"},
                messages=[{"role": "user", "content": f"URL: {page.url}\nTitle: {page.title}\n\n{page.markdown[:MAX_PAGE_CHARS]}"}],
            )
            facts = next((b.input.get("facts", []) for b in resp.content if b.type == "tool_use"), [])
        except Exception as exc:  # network / quota: skip this page, keep going
            with self.lock:
                self.failed += 1
            print(f"  extract failed for {page.url}: {exc}", file=sys.stderr)
            return []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(facts))
        with self.lock:
            self.calls += 1
        return facts


@dataclass
class LLMFact:
    """One LLM-extracted fact with provenance."""
    entity: str
    attribute: str
    value: str
    qualifiers: str
    effective: str
    status: str
    quote: str
    url: str
    snapshot_id: str
    fetched_at: str
    tokens: frozenset
    dated: str = ""  # publication date of the source page; the tie-break when sources disagree


class LLMFactIndex:
    """LLM facts of one snapshot with idf-weighted token-recall lookup."""

    MIN_RECALL = 0.6

    def __init__(self, facts):
        self.facts = facts
        df = {}
        for f in facts:
            for t in f.tokens:
                df[t] = df.get(t, 0) + 1
        n = max(len(facts), 1)
        self.idf = {t: T.math.log(1 + n / c) for t, c in df.items()}
        self.max_idf = max(self.idf.values(), default=1.0)

    def lookup(self, question, limit=4):
        """Return the facts that best explain the question (current, newest first), or []."""
        q = [t for t in toks(question) if t not in T.GENERIC and t not in QUESTION_NOISE]
        if not q or not self.facts:
            return []
        eligibility = "eligib" in q  # who qualifies: prefer the most specific fact over a newer, longer one
        w = {t: self.idf.get(t, self.max_idf) for t in q}
        total = sum(w.values())
        scored = []
        for f in self.facts:
            if f.status == "expired":
                continue
            matched = sum(w[t] for t in q if t in f.tokens or (eligibility and t == "member" and "membership" in f.tokens))
            recall = matched / total
            if recall >= self.MIN_RECALL:
                precision = matched / (sum(self.idf[t] for t in f.tokens) or 1.0)
                scored.append((round(recall, 2), (round(precision, 1), f.dated) if eligibility else (f.dated, 0), f))
        if not scored:
            return []
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", question.lower()):
            scored = [x for x in scored if re.search(r"\d", x[2].value)] or scored  # a count question wants a number
        best = max(s[0] for s in scored)
        top = [(r, d, f) for r, d, f in scored if r >= best - 0.15]
        top.sort(key=lambda x: (x[0], x[1]), reverse=True)  # best recall, then newest article
        seen, out = set(), []
        for _, _, f in top:  # same entity+attribute+qualifiers: keep the most recently fetched
            key = (f.entity.lower(), f.attribute.lower(), f.qualifiers.lower(), f.value.lower())
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out[:limit]


class LLMFactStore:
    """KnowledgeSystem: table facts + LLM facts + BM25 fallback, per snapshot."""

    def __init__(self):
        self.tables = T.FactStore()  # holds SnapshotStore per snapshot (tables + paragraphs)
        self.llm_indexes = []  # parallel to self.tables.stores
        self.extractor = Extractor()
        self.started = time.time()

    def ingest(self, snapshot: Snapshot) -> None:
        """Parse tables, then LLM-extract prose pages (cached, parallel, time-budgeted)."""
        self.tables.ingest(snapshot)
        pages = [p for p in snapshot.pages if wants_llm(p)]
        if self.extractor.enabled or CACHE_DIR.exists():
            def run(page):
                if time.time() - self.started > BUDGET_S and not cache_path(page_hash(page)).exists():
                    return page, []  # out of time budget: only use what is cached
                return page, self.extractor.extract(page)

            with ThreadPoolExecutor(WORKERS) as pool:
                results = list(pool.map(run, pages))
        else:
            results = []
        facts = []
        for page, items in results:
            for it in items:
                try:
                    text = " ".join([it["entity"], it["attribute"], it.get("qualifiers") or ""])
                    when = page_date(page)
                    facts.append(
                        LLMFact(
                            entity=it["entity"], attribute=it["attribute"], value=str(it["value"]),
                            qualifiers=it.get("qualifiers") or "", effective=it.get("effective") or "",
                            status=it.get("status") or "current", quote=it.get("quote") or "",
                            url=page.url, snapshot_id=snapshot.snapshot_id, fetched_at=page.fetched_at,
                            tokens=frozenset(toks(text)), dated=when,
                        )
                    )
                except (KeyError, TypeError):
                    continue
        self.llm_indexes.append(LLMFactIndex(facts))
        e = self.extractor
        print(f"  llm facts: {len(facts)} (calls={e.calls} cached={e.cached} failed={e.failed} enabled={e.enabled})", file=sys.stderr)

    def _pick(self, as_of):
        """Index of the snapshot current at `as_of` (newest when None), or None."""
        stores = self.tables.stores
        if not stores:
            return None
        if not as_of:
            return len(stores) - 1
        ok = [i for i, s in enumerate(stores) if s.taken_on <= as_of]
        return ok[-1] if ok else None

    def answer(self, question, as_of=None):
        """Table lookup, then LLM-fact lookup, then paragraph search, else 'I don't know'."""
        i = self._pick(as_of)
        if i is None:
            return Answer(text="I don't know.", confident=False)
        store = self.tables.stores[i]
        rate_intent = bool(T.RATE_INTENT & set(T.tokenize(question, keep_stop=True)))

        rows, _ = store.lookup(question)
        if rows:
            return self.tables.answer(question, as_of)

        # "Does X pay 0.35%?": the number is a premise to check, not a word to match.
        asked = [float(x) for x in PERCENT.findall(question)]
        if asked and rate_intent:
            rows, _ = store.lookup(PERCENT.sub(" ", question))
            if rows:
                metric = next((m for m in ("apy", "rate", "apr") if any(m in r.values for r in rows)), None)
                rows = [r for r in rows if metric in r.values]
                # "savings tiers" means the plain product: fewest topic words the question did not say, then most tiers
                qt = set(T.tokenize(question))
                extra = lambda r: len(set(T.tokenize(r.topic)) - qt)
                tiers = {}
                for r in rows:
                    tiers.setdefault(r.topic, []).append(r)
                best_topic = min(tiers, key=lambda t: (extra(tiers[t][0]), -len(tiers[t])))
                rows = tiers[best_topic]
                if metric and not any(abs(r.values[metric] - a) < 1e-9 for r in rows for a in asked):
                    top = max(r.values[metric] for r in rows)
                    return Answer(
                        text=f"No - the published {metric.upper()} tops out at {top:.2f}%; {asked[0]:g}% is not on the rate sheet.\n"
                        + "\n".join(T.render(r) for r in rows[:6]),
                        sources=[Source(url=rows[0].url, snapshot_id=store.snapshot_id, quote=T.render(rows[0])[:200])],
                    )

        facts = self.llm_indexes[i].lookup(question)
        if facts:
            if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", question.lower()):
                facts = facts[:1]  # a count has one value; extra lines only add conflicting numbers
            lines = [f"{f.entity} - {f.attribute}{' (' + f.qualifiers + ')' if f.qualifiers else ''}: {f.value}" for f in facts]
            if any("eligib" in f.tokens for f in facts):  # eligibility hinges on the charter area: add that page's geography
                lines += [f"{c.entity} - {c.attribute}: {c.value}" for c in self.llm_indexes[i].facts
                          if c.url == facts[0].url and c not in facts and re.search(r"Iowa|Illinois|Wisconsin", c.value)][:2]
            if facts[0].quote:
                lines.append(f'"{facts[0].quote}"')
            return Answer(
                text="\n".join(lines),
                sources=[Source(url=f.url, snapshot_id=store.snapshot_id, quote=f.quote[:200]) for f in facts[:2]],
            )
        if rate_intent:
            return Answer(text="I don't know. The website doesn't publish that rate; please contact Dupaco directly.", confident=False)
        hit = store.best_paragraph(question)
        if not hit:
            return Answer(text="I don't know.", confident=False)
        url, para = hit
        return Answer(text=para[:600], sources=[Source(url=url, snapshot_id=store.snapshot_id, quote=para[:200])])

    def conflicts(self):
        """Table-row conflicts, plus LLM facts with the same entity+attribute+qualifiers but different values."""
        out = self.tables.conflicts()
        if not self.llm_indexes:
            return out
        groups = {}
        for f in self.llm_indexes[-1].facts:
            if f.status == "current":
                groups.setdefault((f.entity.lower(), f.attribute.lower(), f.qualifiers.lower()), []).append(f)
        for (entity, attr, qual), fs in groups.items():
            by_value = {}
            for f in fs:
                by_value.setdefault(f.value.strip().lower(), f)
            if len(by_value) < 2:
                continue
            win = max(by_value.values(), key=lambda f: (f.dated, f.fetched_at))
            out.append(
                Conflict(
                    topic=f"{entity} {attr}" + (f" ({qual})" if qual else ""),
                    values=[f.value for f in by_value.values()],
                    sources=[Source(url=f.url, snapshot_id=f.snapshot_id, quote=f.quote[:200]) for f in by_value.values()],
                    resolution=win.value,
                    reason="most recently published page",
                )
            )
        return out


def build():
    """Create the knowledge system used by the harness."""
    return LLMFactStore()
