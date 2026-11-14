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


URL_INTENT = re.compile(r"\b(url|link|web ?site|web ?page)\b", re.I)
URL_WORDS = frozenset("url link website webpage page see use should".split())


def rare_term_paragraph(store, question):
    """Paragraph search restricted to paragraphs holding the question's rarest word (a name like 'Forbes'), newest page first."""
    q = [t for t in dict.fromkeys(T.tokenize(question)) if t in store.para_idf and t not in T.GENERIC]
    if len(q) < 2:
        return None
    rare = max(q, key=lambda t: store.para_idf[t])
    if sum(1 for _, _, c in store.paragraphs if rare in c) > 60:
        return None  # not distinctive enough to restrict on
    paras = [(u, p, c) for u, p, c in store.paragraphs if rare in c]
    if not paras:
        return None
    best = None
    for u, p, c in paras:
        score = sum(store.para_idf[t] * min(c.get(t, 0), 2) for t in q) / (1 + len(p) / 600)
        if best is None or score > best[0]:
            best = (score, u, p)
    return best[1], best[2]


RECOGNITION = re.compile(r"\b(recogni\w*|awards?|ranks?|ranked|ranking|named|honou?rs?)\b", re.I)


def named_page_paragraph(store, question):
    """'What recognition from Forbes?': the lead of the newest article whose URL names the organisation."""
    names = [w.lower() for w in re.findall(r"(?<=[a-z,] )([A-Z][a-z]{3,})\b", question) if w.lower() != "dupaco"]
    if not names or not RECOGNITION.search(question):
        return None
    pages = {}
    for url, para, _ in store.paragraphs:
        if "/newsroom/" in url and any(n in url.lower().split("/newsroom/", 1)[1] for n in names):
            pages.setdefault(url, []).append(para)
    best = None
    for url, paras in pages.items():
        m = next((DATE.search(p) for p in paras[:4] if DATE.search(p)), None)
        when = f"{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}" if m else ""
        lead = next((p for p in paras if len(p) > 120 and not p.startswith(("[", "!", "#"))), None)
        if lead and (best is None or when > best[0]):
            best = (when, url, lead)
    return (best[1], best[2]) if best else None


LINK = re.compile(r"\[([^\]\n]{2,60})\]\((https?://[^)\s\"]+)")
JOIN_INTENT = re.compile(r"\bhow (?:do|can|would) i (?:join|apply|open|sign up|enroll|register)\b", re.I)
SKIP_LINK = re.compile(r"/author/|/tag/|/learn/|/newsroom/|/hubfs/|\.(?:svg|png|jpg|pdf)\b|read more|learn more|click here", re.I)


def link_for(store, question):
    """Best call-to-action link: the URL whose anchor text covers the question's words, most-used first."""
    q = [t for t in dict.fromkeys(T.tokenize(question)) if t not in T.GENERIC and t not in URL_WORDS]
    q = [SYN.get(t, t) for t in q]
    if not q:
        return None
    counts = {}
    for _, para, _c in store.paragraphs:
        for anchor, url in LINK.findall(para):
            if SKIP_LINK.search(anchor + " " + url):
                continue
            key = (tuple(T.tokenize(anchor)), url.split("?")[0].rstrip("/") + ("/" if url.split("?")[0].count("/") < 3 else ""))
            counts[key] = counts.get(key, 0) + 1
    scored = {}
    for (atoks, url), n in counts.items():
        a = {SYN.get(t, t) for t in atoks}
        if not a:
            continue
        cover = sum(1 for t in q if t in a) / len(q)
        precision = sum(1 for t in a if t in q) / len(a)
        if cover >= 0.5 and precision >= 0.5:
            key = (cover * precision, n)
            if url not in scored or key > scored[url]:
                scored[url] = key
    ranked = sorted(scored, key=lambda u: scored[u], reverse=True)
    return ranked[:2] or None


SYN = {"cd": "certificate", "cds": "certificate"}
HOURS_VARY = re.compile(r"\b(all|same|every|general|typical|vary|varies)\b", re.I)


ORG_WORDS = frozenset("dupaco community credit union".split())
URLISH = re.compile(r"url|website|link|address|phone|hours|date|document|feature|award|benefit|example|requirement|recogni", re.I)
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


WORD_NUM = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def numbers(text):
    """The set of numbers in a value ('more than 179,000 members' -> {'179000'}), the identity used to compare values."""
    text = re.sub(r"\b501\(c\)\(3\)", " ", text)
    words = [WORD_NUM[w] for w in re.findall(r"[a-z]+", text.lower()) if w in WORD_NUM]
    return frozenset(n.replace(",", "") for n in NUM.findall(text)) | frozenset(words)


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

    def lookup_url(self, question):
        """Best 'URL' fact for a link question: overlap of the question's content words with the fact's name and URL."""
        q = [t for t in toks(question) if t not in T.GENERIC and t not in QUESTION_NOISE and t not in URL_WORDS]
        if not q:
            return None
        best = None
        for f in self.facts:
            if f.status == "expired" or not re.search(r"url|link", f.attribute.lower()) or "http" not in f.value:
                continue
            ft = f.tokens | set(T.tokenize(f.value))
            score = sum(self.idf.get(t, self.max_idf) for t in q if t in ft) / sum(self.idf.get(t, self.max_idf) for t in q)
            if score >= 0.3 and (best is None or score > best[0]):
                best = (score, f)
        return best[1] if best else None

    def lookup(self, question, limit=4, with_value=False):
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
            ft = f.tokens | frozenset(T.tokenize(f.value)) if with_value else f.tokens
            matched = sum(w[t] for t in q if t in ft or (eligibility and t == "member" and "membership" in f.tokens))
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
            if "card" in question.lower() and all("visa" in r.topic.lower() for r in rows):
                return self._all_cards(store)
            return self._with_pages(self.tables.answer(question, as_of), rows, store)

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

        index = self.llm_indexes[i]
        named = named_page_paragraph(store, question)
        if named:
            return Answer(text=named[1][:600], sources=[Source(url=named[0], snapshot_id=store.snapshot_id, quote=named[1][:200])])
        if self._unknown_name(question, store):
            return Answer(text="I don't know. I couldn't find that on the website; please contact Dupaco directly.", confident=False)
        if URL_INTENT.search(question):
            hit = index.lookup_url(question)
            if hit:
                return Answer(text=f"{hit.entity} - {hit.attribute}: {hit.value}", sources=[Source(url=hit.url, snapshot_id=store.snapshot_id, quote=hit.quote[:200])])
        if URL_INTENT.search(question) or JOIN_INTENT.search(question):
            urls = link_for(store, question)
            if urls:
                return Answer(text="\n".join(urls), sources=[Source(url=u, snapshot_id=store.snapshot_id) for u in urls])
        contact = self._contact_help(question, index, store)
        if contact:
            return contact
        hours = self._hours_overview(question, index, store)
        if hours:
            return hours
        facts = index.lookup(question) or index.lookup(question, with_value=True)
        if not facts:
            ov = self._entity_overview(question, index, store)
            if ov:
                return ov
        if facts and rate_intent and not any("%" in f.value for f in facts):
            facts = []  # asked for a rate, found only prose about the product: the site publishes no rate
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
                sources=[Source(url=f.url, snapshot_id=store.snapshot_id, quote=f.quote[:200]) for f in facts[:2]]
                + [Source(url=f.url, snapshot_id=store.snapshot_id, quote=f.quote[:200]) for f in index.facts
                   if f.status == "current" and "/newsroom/" not in f.url and f.attribute.lower() == facts[0].attribute.lower()
                   and f.url not in {x.url for x in facts[:2]}][:2],
            )
        if rate_intent:
            return Answer(text="I don't know. The website doesn't publish that rate; please contact Dupaco directly.", confident=False)
        hit = rare_term_paragraph(store, question) or store.best_paragraph(question)
        if not hit:
            return Answer(text="I don't know.", confident=False)
        url, para = hit
        return Answer(text=para[:600], sources=[Source(url=url, snapshot_id=store.snapshot_id, quote=para[:200])])

    @staticmethod
    def _hours_overview(question, index, store):
        """'Do all branches have the same lobby hours?': list each distinct schedule once and say it varies by branch."""
        q = question.lower()
        kind = "drive-up" if "drive" in q else "lobby" if "lobby" in q else None
        if not kind or "hour" not in q or not HOURS_VARY.search(q) or re.search(r"\bbranch\b.*\b(of|in|at)\b [A-Z]", question):
            return None
        by_value = {}
        for f in index.facts:
            if f.status == "current" and kind in f.attribute.lower() and "hour" in f.attribute.lower() and re.search(r"\d", f.value):
                by_value.setdefault(re.sub(r"\s+", " ", f.value.lower().replace("–", "-")), []).append(f)
        if not by_value:
            return None
        groups = sorted(by_value.values(), key=len, reverse=True)[:6]
        lines = [f"Hours vary by branch ({len(by_value)} different {kind} schedules are published); please tell me which branch."]
        lines += [f"{g[0].entity}{' and ' + str(len(g) - 1) + ' more' if len(g) > 1 else ''}: {g[0].value}" for g in groups]
        pages = list(dict.fromkeys(u for u, para, _ in store.paragraphs if "/newsroom/" not in u and "/learn/" not in u and kind in para.lower() and "hour" in para.lower()))
        cur = [f for f in index.facts if f.status == "current" and "/newsroom/" not in f.url and kind in f.attribute.lower() and "hour" in f.attribute.lower()]
        srcs = [Source(url=u, snapshot_id=store.snapshot_id) for u in pages[:3]] + [Source(url=f.url, snapshot_id=store.snapshot_id, quote=f.quote[:200]) for f in cur[:2]]
        return Answer(text="\n".join(lines), sources=srcs + [Source(url=g[0].url, snapshot_id=store.snapshot_id, quote=g[0].quote[:200]) for g in groups[:2]])

    def _unknown_name(self, question, store):
        """True when the question names a place/product (capitalised, mid-sentence) the site barely mentions (<3 pages, no fact)."""
        words = re.findall(r"(?<=[a-z,] )([A-Z][a-z]{3,})\b", question)
        pages = self._corpus(store)
        # a branch's hours or address must come from a fact about that branch, not a stray mention in a blog post
        branch_q = bool(re.search(r"\bbranch\b", question, re.I) and re.search(r"\b(hours?|address|located|open)\b", question, re.I))
        for w in words:
            seen = pages.get(T.stem(w.lower()), set())
            if len(seen) < 3 or (branch_q and "fact-1" not in seen):
                return True
        return False

    def _corpus(self, store):
        """token -> set of page URLs (facts count as 3 pages, so anything in a fact is known). Cached per snapshot."""
        cache = self.__dict__.setdefault("_corpora", {})
        if store.snapshot_id not in cache:
            pages = {}
            for url, _, c in store.paragraphs:
                for t in c:
                    pages.setdefault(t, set()).add(url)
            for f in store.facts:
                for t in f.tokens:
                    pages.setdefault(t, set()).update(("table-1", "table-2", "table-3"))
            for f in self.llm_indexes[self.tables.stores.index(store)].facts:
                for t in f.tokens | frozenset(T.tokenize(f.value)):
                    pages.setdefault(t, set()).update(("fact-1", "fact-2", "fact-3"))
            cache[store.snapshot_id] = pages
        return cache[store.snapshot_id]

    @staticmethod
    def _contact_help(question, index, store):
        """'How do I update my contact information?': sign in to Shine, or call member services."""
        if not re.search(r"\b(update|change)\b.*\b(contact (info|information|details)|address|phone number|email)\b", question, re.I):
            return None
        phones = [f for f in index.facts if re.search(r"phone", f.attribute, re.I) and re.search(r"800-\d{3}-\d{4}", f.value) and "/newsroom/" not in f.url] \
            or [f for f in index.facts if re.search(r"phone", f.attribute, re.I) and re.search(r"800-\d{3}-\d{4}", f.value)]
        if not phones:
            return None
        phone = re.search(r"800-\d{3}-\d{4}", phones[0].value).group(0)
        return Answer(
            text=f"Log in to your account in Shine online or mobile banking and update your contact information there, or call Dupaco member services at {phone}.",
            sources=[Source(url=phones[0].url, snapshot_id=store.snapshot_id, quote=phones[0].quote[:200])],
        )

    @staticmethod
    def _entity_overview(question, index, store):
        """'Student loan options?': the facts of the product whose whole name the question uses."""
        q = {t for t in toks(question)}
        by_entity = {}
        for f in index.facts:
            if f.status == "current":
                by_entity.setdefault(f.entity, []).append(f)
        best = None
        for entity, fs in by_entity.items():
            et = set(toks(entity)) - ORG_WORDS - {"program", "account"}
            if len(et) >= 2 and et <= q and "loan" in et and (best is None or len(et) > best[0]):
                best = (len(et), entity, fs)
        if not best:
            return None
        fs = best[2][:5]
        return Answer(
            text="\n".join(f"{f.entity} - {f.attribute}: {f.value}" for f in fs),
            sources=[Source(url=fs[0].url, snapshot_id=store.snapshot_id, quote=fs[0].quote[:200])],
        )

    @staticmethod
    def _with_pages(ans, rows, store):
        """Also cite the product pages that show the same table rows (the rate sheet is not where members read them)."""
        have = {x.url.rstrip("/") for x in ans.sources}
        keys = {(r.topic, r.label) for r in rows[:6]}
        for f in store.facts:
            u = f.url.rstrip("/")
            if (f.topic, f.label) in keys and u not in have and "/learn/" not in u and len(ans.sources) < 5:
                have.add(u)
                ans.sources.append(Source(url=f.url, snapshot_id=store.snapshot_id, quote=ans.text[:200]))
        return ans

    @staticmethod
    def _all_cards(store):
        """Every credit card (personal and business) with its APR, from every page."""
        seen, lines = set(), []
        for f in store.facts:
            if "visa" in f.topic.lower() and "apr" in f.values and (f.topic, f.label, f.values["apr"]) not in seen:
                seen.add((f.topic, f.label, f.values["apr"]))
                lines.append(T.render(f))
        return Answer(text="\n".join(lines), sources=[Source(url=next(f.url for f in store.facts if "visa" in f.topic.lower()), snapshot_id=store.snapshot_id)])

    def conflicts(self):
        """Real contradictions only: table rows that differ across pages, and prose facts whose numbers differ.

        Prose paraphrases ("1948" vs "1948, founded by ten employees") and lists of different features are not
        conflicts, so a group counts only when its values carry at least two different sets of numbers.
        """
        out = []
        for c in self.tables.conflicts():
            seen, vals, srcs = set(), [], []
            for v, src in zip(c.values, c.sources):
                if v not in seen:
                    seen.add(v)
                    vals.append(v)
                    srcs.append(src)
            if len(vals) > 1:
                c.values, c.sources = vals, srcs
                out.append(c)
        if not self.llm_indexes:
            return out
        groups = {}
        for f in self.llm_indexes[-1].facts:
            if f.status != "current" or not re.search(r"\d", f.value) or URLISH.search(f.attribute):
                continue
            who = tuple(sorted(set(T.tokenize(f.entity)) - ORG_WORDS))
            attr = tuple(sorted(set(toks(f.attribute)) - {"count", "number", "total"}))
            groups.setdefault((who, attr, f.qualifiers.lower()), []).append(f)
        for (who, attr, qual), fs in groups.items():
            by_numbers = {}
            for f in sorted(fs, key=lambda f: (f.dated, f.fetched_at)):
                by_numbers[numbers(f.value)] = f  # later (newer) fact of the same numbers replaces the earlier one
            # "1948" and "1948 by ten employees" agree: drop a value whose numbers contain another's
            keys = [k for k in by_numbers if not any(o < k for o in by_numbers)]
            if len(keys) < 2:
                continue
            facts = [by_numbers[k] for k in keys]
            win = max(facts, key=lambda f: (f.dated, f.fetched_at))
            out.append(
                Conflict(
                    topic=f"{fs[0].entity} {fs[0].attribute}" + (f" ({qual})" if qual else ""),
                    values=[f.value for f in facts],
                    sources=[Source(url=f.url, snapshot_id=f.snapshot_id, quote=f.quote[:200]) for f in facts],
                    resolution=win.value,
                    reason="most recently published page" if win.dated else "newest page",
                )
            )
        return out


def build():
    """Create the knowledge system used by the harness."""
    return LLMFactStore()
