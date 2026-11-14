"""
Knowledge graph with time validity (Approach 5).

Graph
    Nodes   product        a table topic, e.g. "Regular Certificates"
            tier row       a row of that table, e.g. "12 Month" or "2024-27 Model Year | 60 Months"
            entity         an LLM-extracted subject, e.g. "Hy-Vee branch"
    Edges   product --has_row--> tier row --has_metric--> (rate | apy | apr)
            entity  --has_attribute--> value
    Each metric / attribute is a TIMELINE of versions:
            Version(value, valid_from, valid_to, source)
    valid_from = date of the first snapshot where the value appeared.
    valid_to   = date of the first snapshot where it was gone or different (exclusive), None while it
                 still holds in the newest snapshot. Real change dates lie between two crawls, so
                 the graph only knows "changed between D1 and D2".

Query
    answer(q, as_of): rows/facts are matched exactly as in the table / LLM-fact layers, but the value
    returned is the version valid at `as_of` (newest snapshot when None). Only when the member asks about
    change ("did it change", "previous rate", "history") is the timeline rendered. Entities that existed
    earlier but are gone from the newest snapshot are reported as no longer listed.

conflicts()
    (a) same row, different values on different pages in the newest snapshot (official rate sheet wins),
    (b) values that were superseded between snapshots: old value -> new value with the dates.

Setup: table facts need nothing. Prose facts come from the LLM extraction cache in kg_time/cache (copied
from llm_facts); with ANTHROPIC_API_KEY and `pip install anthropic`, uncached pages are extracted too.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
import llm_layer as L  # noqa: E402
import table_facts as T  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

CHANGE_WORDS = frozenset(
    "change changed changes changing previous previously before earlier history historical trend increase "
    "increased decrease decreased rise rose raised drop dropped lowered used went old".split()
)


T.GENERIC = T.GENERIC | CHANGE_WORDS | frozenset("how".split())  # change words say nothing about WHICH product


@dataclass
class Version:
    """One value of a metric / attribute and the span of snapshots it held for."""
    value: object
    from_idx: int
    from_date: str
    to_idx: int = None
    to_date: str = None
    url: str = ""


@dataclass
class Node:
    """A node of the graph with its timelines."""
    kind: str  # product | row | entity
    name: str
    parent: str = None
    timelines: dict = field(default_factory=dict)  # metric/attribute -> [Version]


class TimeGraph:
    """Nodes + timelines, updated one snapshot at a time (oldest first)."""

    def __init__(self):
        self.nodes = {}
        self.dates = []  # snapshot dates by index

    def node(self, key, kind, name, parent=None):
        """Get or create a node."""
        if key not in self.nodes:
            self.nodes[key] = Node(kind, name, parent)
        return self.nodes[key]

    def observe(self, idx, date, observations):
        """Apply one snapshot: observations = {(node_key, metric): (value, url)}. Closes vanished / changed versions."""
        for key, node in self.nodes.items():
            for metric, versions in node.timelines.items():
                last = versions[-1]
                if last.to_idx is None and (key, metric) not in observations:
                    last.to_idx, last.to_date = idx, date  # disappeared
        for (key, metric), (value, url) in observations.items():
            versions = self.nodes[key].timelines.setdefault(metric, [])
            last = versions[-1] if versions else None
            if last and last.to_idx is None and last.value == value:
                last.url = url
                continue
            if last and last.to_idx is None:
                last.to_idx, last.to_date = idx, date  # superseded
            versions.append(Version(value, idx, date, url=url))

    @staticmethod
    def valid_at(versions, idx):
        """The version whose span covers snapshot `idx`, or None."""
        for v in versions:
            if v.from_idx <= idx and (v.to_idx is None or idx < v.to_idx):
                return v
        return None


def row_key(f):
    """Row identity that survives crawls: personal/business, label text, and which metrics the row carries."""
    metrics = tuple(m for m in ("rate", "apy", "apr") if m in f.values)
    structural = (f.values.get("min_open"), f.values.get("min_earn"))  # minimums do not move with rates
    return ("row", "business" in f.url, f.label.lower(), metrics, structural)


def numbers(value):
    """Sorted digit groups of a text value, to tell real changes from rewording."""
    return sorted(T.re.findall(r"\d[\d,.]*", str(value).replace(",", "")))


def fmt(metric, value):
    """Format a metric value for text."""
    return f"{metric.upper()} {value:.2f}%" if metric in ("apy", "apr") else (f"rate {value:.2f}%" if metric == "rate" else str(value))


def span(v):
    """Human-readable validity span of a version."""
    end = v.to_date if v.to_idx is not None else "now"
    return f"{v.from_date} to {end}"


class TimeKnowledgeGraph:
    """KnowledgeSystem backed by a time-validity graph (+ BM25 fallback)."""

    def __init__(self):
        self.tables = T.FactStore()
        self.graph = TimeGraph()
        self.extractor = L.Extractor()
        self.llm_indexes = []
        self.fact_keys = []  # per snapshot: {id(fact): node_key}
        self.started = L.time.time()

    # ---- ingest ------------------------------------------------------------
    def ingest(self, snapshot: Snapshot) -> None:
        """Parse tables + extract prose facts, then update the graph timelines for this snapshot."""
        self.tables.ingest(snapshot)
        store = self.tables.stores[-1]
        idx = len(self.tables.stores) - 1
        self.graph.dates.append(snapshot.taken_on)

        # LLM prose facts (cache first; API only if a key is configured)
        pages = [p for p in snapshot.pages if L.wants_llm(p)]
        facts = []
        if self.extractor.enabled or L.CACHE_DIR.exists():
            def run(page):
                if L.time.time() - self.started > L.BUDGET_S and not L.cache_path(L.page_hash(page)).exists():
                    return page, []
                return page, self.extractor.extract(page)

            with ThreadPoolExecutor(L.WORKERS) as pool:
                for page, items in pool.map(run, pages):
                    for it in items:
                        try:
                            text = " ".join([it["entity"], it["attribute"], it.get("qualifiers") or ""])
                            facts.append(L.LLMFact(
                                entity=it["entity"], attribute=it["attribute"], value=str(it["value"]),
                                qualifiers=it.get("qualifiers") or "", effective=it.get("effective") or "",
                                status=it.get("status") or "current", quote=it.get("quote") or "",
                                url=page.url, snapshot_id=snapshot.snapshot_id, fetched_at=page.fetched_at,
                                tokens=frozenset(T.tokenize(text)),
                            ))
                        except (KeyError, TypeError):
                            continue
        self.llm_indexes.append(L.LLMFactIndex(facts))

        # Table rows: one winner per (section, topic, label): official page, then newest fetch
        obs, winners = {}, {}
        for f in store.facts:
            key = row_key(f)
            if key not in winners or (f.priority, f.fetched_at) > (winners[key].priority, winners[key].fetched_at):
                winners[key] = f
        for key, f in winners.items():
            pkey = ("product", f.topic.lower() or "unlabelled")
            self.graph.node(pkey, "product", f.topic or "unlabelled")
            row = self.graph.node(key, "row", f.label, parent=pkey)
            row.parent = pkey if f.topic else row.parent  # later crawls name the product; keep the latest
            for metric in ("rate", "apy", "apr"):
                if metric in f.values:
                    obs[(key, metric)] = (f.values[metric], f.url)
        # LLM facts: one winner per (entity, attribute, qualifiers)
        ewin = {}
        for f in facts:
            if f.status == "expired":
                continue
            key = ("entity", f.entity.lower(), f.attribute.lower(), f.qualifiers.lower())
            if key not in ewin or f.fetched_at > ewin[key].fetched_at:
                ewin[key] = f
        for key, f in ewin.items():
            self.graph.node(key, "entity", f.entity)
            obs[(key, f.attribute.lower())] = (f.value, f.url)
        self.graph.observe(idx, snapshot.taken_on, obs)
        self.fact_keys.append({"rows": winners})
        e = self.extractor
        print(f"  graph nodes: {len(self.graph.nodes)} (llm facts {len(facts)}, calls={e.calls} cached={e.cached})", file=sys.stderr)

    # ---- query -------------------------------------------------------------
    def _pick(self, as_of):
        """Snapshot index current at `as_of` (newest when None), or None."""
        stores = self.tables.stores
        if not stores:
            return None
        if not as_of:
            return len(stores) - 1
        ok = [i for i, s in enumerate(stores) if s.taken_on <= as_of]
        return ok[-1] if ok else None

    def answer(self, question, as_of=None):
        """Match like the table/LLM layers, return the version valid at as_of; render history on change questions."""
        i = self._pick(as_of)
        if i is None:
            return Answer(text="I don't know.", confident=False)
        store = self.tables.stores[i]
        wants_history = bool(CHANGE_WORDS & set(T.tokenize(question, keep_stop=True))) and not as_of
        rows, _ = store.lookup(question)
        if rows:
            if not wants_history:
                return self.tables.answer(question, as_of)
            return self._history_answer(rows, i, store)
        facts = self.llm_indexes[i].lookup(question)
        if facts:
            lines = []
            for f in facts:
                key = ("entity", f.entity.lower(), f.attribute.lower(), f.qualifiers.lower())
                line = f"{f.entity} - {f.attribute}{' (' + f.qualifiers + ')' if f.qualifiers else ''}: {f.value}"
                versions = self.graph.nodes.get(key, Node("entity", "")).timelines.get(f.attribute.lower(), [])
                if wants_history and len(versions) > 1:
                    line += " | history: " + "; ".join(f"{v.value} ({span(v)})" for v in versions)
                lines.append(line)
            if facts[0].quote:
                lines.append(f'"{facts[0].quote}"')
            return Answer(
                text="\n".join(lines),
                sources=[Source(url=f.url, snapshot_id=store.snapshot_id, quote=f.quote[:200]) for f in facts[:2]],
            )
        if T.RATE_INTENT & set(T.tokenize(question, keep_stop=True)):
            return Answer(text="I don't know. The website doesn't publish that rate; please contact Dupaco directly.", confident=False)
        hit = store.best_paragraph(question)
        if not hit:
            return Answer(text="I don't know.", confident=False)
        url, para = hit
        return Answer(text=para[:600], sources=[Source(url=url, snapshot_id=store.snapshot_id, quote=para[:200])])

    def _history_answer(self, rows, idx, store):
        """Current value plus the timeline of each matched row."""
        lines, urls = [], []
        for f in rows:
            key = row_key(f)
            node = self.graph.nodes.get(key)
            name = f"{f.topic}: {f.label}"
            parts = []
            for metric in ("rate", "apy", "apr"):
                versions = node.timelines.get(metric, []) if node else []
                cur = self.graph.valid_at(versions, idx)
                if not cur:
                    continue
                part = f"{fmt(metric, cur.value)} (since {cur.from_date})"
                older = [v for v in versions if v.from_idx < cur.from_idx]
                if older:
                    part += ", before: " + ", ".join(f"{fmt(metric, v.value)} ({span(v)})" for v in older)
                else:
                    part += ", unchanged across crawls"
                parts.append(part)
            lines.append(f"{name}: " + "; ".join(parts))
            urls.append(f.url)
        lines.append("Dates are crawl dates; the exact change day falls between two crawls.")
        return Answer(
            text="\n".join(lines),
            sources=[Source(url=urls[0], snapshot_id=store.snapshot_id)],
        )

    def history(self, topic_words):
        """Debug helper: timelines of every row/entity whose name contains all words."""
        words = [w.lower() for w in topic_words.split()]
        out = {}
        for key, node in self.graph.nodes.items():
            if all(w in " ".join(str(k) for k in key).lower() for w in words) and node.timelines:
                out[key] = {m: [(v.value, span(v)) for v in vs] for m, vs in node.timelines.items()}
        return out

    # ---- conflicts ---------------------------------------------------------
    def conflicts(self):
        """Same-snapshot disagreements between pages, plus values superseded across snapshots."""
        out = list(self.tables.conflicts())
        latest = len(self.tables.stores) - 1
        for key, node in self.graph.nodes.items():
            for metric, versions in node.timelines.items():
                if len(versions) < 2 or versions[-1].to_idx is not None:
                    continue  # unchanged, or no longer current (nothing to resolve)
                new = versions[-1]
                olds, seen = [], {new.value}
                for v in versions[:-1]:
                    if v.value not in seen and numbers(v.value) != numbers(new.value):  # skip pure rewording
                        seen.add(v.value)
                        olds.append(v)
                if not olds:
                    continue
                parent = self.graph.nodes.get(node.parent)
                topic = f"{parent.name + ' ' if parent else ''}{node.name} {metric.upper()}".strip()
                shown = lambda v: fmt(metric, v.value) if isinstance(v.value, float) else str(v.value)
                out.append(Conflict(
                    topic=topic,
                    values=[shown(v) for v in olds] + [shown(new)],
                    sources=[Source(url=v.url, snapshot_id=v.from_date) for v in olds + [new]],
                    resolution=shown(new),
                    reason=f"superseded: value changed between crawls (valid since {new.from_date}); older values are stale",
                ))
        return out


def build():
    """Create the knowledge system used by the harness."""
    return TimeKnowledgeGraph()
