"""
Structured fact store: parse tables into facts, parse the question into the same shape, look up.

Ingest
    Every markdown table on every page becomes a list of `Fact`s, one per row:
        topic      table title ("Auto & Motorcycle Loans", "Regular Certificates", ...)
        label      row identity ("2024-27 Model Year | 60 Months")
        values     {"rate": 3.37, "apy": 3.40, "apr": 5.39, "min": 1000.0}
        range      (lo, hi) numeric range found in the label (balance / loan-amount tier), if any
        year range (lo, hi) model-year range found in the label, if any
        source     url, snapshot id, page priority
    Facts are kept per snapshot, so `as_of` picks the snapshot current on that date.

Answer
    The question is parsed into: content tokens, amounts ($), a model year, and intent
    (rate vs. other, highest/lowest, refinance). Rows are scored by how much of the question
    they explain, filtered by tier/year qualifiers, and rendered. If no row explains a rate
    question the answer is an honest "not published". Non-rate questions fall back to BM25
    over paragraphs of the same snapshot.

Conflicts
    Rows with the same identity and metric but different values on different pages of the
    newest snapshot. The official rate sheet (/rates) wins, otherwise the newest page.
"""
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
YEAR_RANGE = re.compile(r"\b((?:19|20)\d\d)\s*[-–]\s*((?:19|20)?\d\d)\b")
YEAR = re.compile(r"\b(19\d\d|20[0-3]\d)\b")

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or the to what when "
    "where which who will with you your me tell give provide get would like know please about "
    "dupaco s".split()
)
# Words that signal "give me a rate" but say nothing about WHICH product.
GENERIC = frozenset(
    "rate rates interest current currently right now today apy apr dividend dividends percent "
    "percentage annual yield much charge pay earn offer offers have there any all list balance between tier account accounts standard".split()
)
RATE_INTENT = frozenset("rate rates interest apy apr dividend yield percentage".split())
SYNONYMS = {
    "mortgage": {"home", "purchase"},
    "cd": {"certificate"},
    "cds": {"certificate"},
    "certificates": {"certificate"},
    "vehicle": {"auto"},
    "car": {"auto"},
    "motorcycle": {"auto"},
    "boat": {"boat"},
    "checking": {"checking"},
    "visa": {"visa"},
    "heloc": {"heloc"},
}
MIN_RECALL = 0.6
MAX_ROWS = 24
OFFICIAL_PRIORITY = {"/rates": 3, "/business-rates": 3}


def stem(word):
    """Crude plural stemmer: 'months' -> 'month', 'rates' -> 'rate'."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text, keep_stop=False):
    """Lowercase, drop thousands commas and dollar amounts, stem. Returns a list of tokens."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = AMOUNT.sub(" ", text)
    return [stem(t) for t in TOKEN.findall(text) if keep_stop or t not in STOPWORDS]


def parse_amount(match):
    """Convert an AMOUNT regex match to a float ($100k -> 100000)."""
    value = float(match.group(1).replace(",", ""))
    return value * 1000 if match.group(2) else value


def parse_range(text):
    """Extract a numeric (lo, hi) tier from a label like '$10,000 – $24,999.99', '$25,000+', '$100k or less'."""
    amounts = [parse_amount(m) for m in AMOUNT.finditer(text)]
    if not amounts:
        return None
    lowered = text.lower()
    if "or less" in lowered or "up to" in lowered:
        return (0.0, amounts[0])
    if "+" in text or "or more" in lowered or "and up" in lowered:
        return (amounts[0], math.inf)
    if len(amounts) >= 2:
        return (min(amounts[:2]), max(amounts[:2]))
    return None


def parse_year_range(text):
    """Extract a model-year range like '2024-27 Model Year' or '2016 Model Year (and older)'."""
    m = YEAR_RANGE.search(text)
    if m:
        lo, hi = int(m.group(1)), m.group(2)
        hi = int(hi) if len(hi) == 4 else int(str(lo)[:2] + hi)
        return (lo, hi)
    if "model year" in text.lower():
        m = YEAR.search(text)
        if m:
            y = int(m.group(1))
            return (0, y) if "older" in text.lower() else (y, y)
    return None


@dataclass
class Fact:
    """One table row, normalized."""
    topic: str
    label: str
    values: dict
    url: str
    snapshot_id: str
    fetched_at: str
    priority: int
    section: str = ""
    tier: tuple = None
    years: tuple = None
    tokens: frozenset = field(default_factory=frozenset)
    raw: dict = field(default_factory=dict)


def classify_header(header):
    """Map a table column header to a metric name, or None if it is a qualifier column."""
    h = header.lower()
    if "apy" in h or "yield" in h:
        return "apy"
    if "apr" in h or "annual percentage rate" in h:
        return "apr"
    if "dividend" in h or h.strip() in {"rate", "rates"}:
        return "rate"
    if "min" in h and "earn" in h:
        return "min_earn"
    if "min" in h and "open" in h:
        return "min_open"
    if "point" in h or "payment" in h:
        return "other"
    return None


def first_percent(cell):
    """Return the first percentage in a cell as a float, ignoring footnote digits ('4.20%1' -> 4.2)."""
    m = PERCENT.search(cell)
    return float(m.group(1)) if m else None


def split_row(line):
    """Split a markdown table row into stripped cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def plain_heading(line):
    """Return the heading text if the line is a short plain heading, else None."""
    s = line.strip()
    if not s or s.startswith(("[", "!", "|", "_", "*", "-")) or len(s) > 80:
        return None
    if s.lower().startswith("rates are effective") or re.match(r"^\d", s) or s.endswith((".", ":")):
        return None
    return s.lstrip("# ").strip()


def table_title(lines, start):
    """Return (title, section): the nearest plain line above the table and the plain line just above that."""
    found = []
    for i in range(start - 1, max(start - 8, -1), -1):
        if lines[i].strip() == "":
            continue
        h = plain_heading(lines[i])
        if h is None:
            break
        found.append(h)
        if len(found) == 2:
            break
    title = found[0] if found else ""
    section = found[1] if len(found) > 1 else None
    return title, section


def extract_facts(page, snapshot_id):
    """Parse every table of a page into Facts."""
    lines = page.markdown.splitlines()
    path = "/" + page.url.split("dupaco.com", 1)[-1].strip("/").split("?")[0].split("#")[0]
    priority = OFFICIAL_PRIORITY.get(path, 1)
    facts, i, section = [], 0, ""
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            title, new_section = table_title(lines, i)
            if new_section:
                section = new_section
            headers = split_row(lines[i])
            kinds = [classify_header(h) for h in headers]
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = split_row(lines[j])
                j += 1
                if len(cells) != len(headers):
                    continue
                values, label_parts = {}, []
                for kind, cell in zip(kinds, cells):
                    if kind is None:
                        if cell and cell != "—":
                            label_parts.append(cell)
                    elif kind in ("rate", "apy", "apr"):
                        v = first_percent(cell)
                        if v is not None:
                            values[kind] = v
                    elif kind in ("min_open", "min_earn"):
                        m = AMOUNT.search(cell)
                        if m:
                            values[kind] = parse_amount(m)
                if not values:
                    continue
                label = " | ".join(dict.fromkeys(label_parts))
                facts.append(
                    Fact(
                        topic=title,
                        label=label,
                        section=section,
                        values=values,
                        url=page.url,
                        snapshot_id=snapshot_id,
                        fetched_at=page.fetched_at,
                        priority=priority,
                        tier=parse_range(label),
                        years=parse_year_range(label),
                        tokens=frozenset(tokenize(f"{section} {title} {label}")),
                        raw=dict(zip(headers, cells)),
                    )
                )
            i = j
        else:
            i += 1
    return facts


class SnapshotStore:
    """Facts and paragraphs of one snapshot, with an idf table for scoring."""

    def __init__(self, snapshot):
        self.snapshot_id = snapshot.snapshot_id
        self.taken_on = snapshot.taken_on
        self.facts = []
        self.paragraphs = []  # (url, text, Counter)
        for page in snapshot.pages:
            self.facts.extend(extract_facts(page, snapshot.snapshot_id))
            for para in page.markdown.split("\n\n"):
                para = para.strip()
                toks = tokenize(para)
                if toks and not para.startswith("|"):
                    self.paragraphs.append((page.url, para, Counter(toks)))
        df = Counter(t for f in self.facts for t in f.tokens)
        n = max(len(self.facts), 1)
        self.fact_idf = {t: math.log(1 + n / c) for t, c in df.items()}
        pdf = Counter(t for _, _, c in self.paragraphs for t in c)
        pn = max(len(self.paragraphs), 1)
        self.para_idf = {t: math.log(1 + (pn - c + 0.5) / (c + 0.5)) for t, c in pdf.items()}
        self.avg_len = sum(sum(c.values()) for _, _, c in self.paragraphs) / pn

    # ---- structured lookup -------------------------------------------------
    def lookup(self, question):
        """Return (facts, recall) best explaining the question, or ([], 0.0)."""
        q_tokens = [t for t in tokenize(question) if t not in GENERIC]
        text = question.lower()
        amounts = [parse_amount(m) for m in AMOUNT.finditer(re.sub(r"(?<=\d),(?=\d)", ",", question))]
        years = [int(y) for y in YEAR.findall(question)]
        year = years[0] if years else None
        q_tokens = [t for t in q_tokens if not (year and t == str(year))]
        if not q_tokens:
            return [], 0.0
        weights = {t: self.fact_idf.get(t, 1.0) for t in q_tokens}
        total = sum(weights.values())

        scored = []
        for f in self.facts:
            matched = 0.0
            for t in q_tokens:
                if t in f.tokens or any(s in f.tokens for s in SYNONYMS.get(t, ())):
                    matched += weights[t]
            if year and f.years and f.years[0] <= year <= f.years[1]:
                matched += max(weights.values())
                total_y = total + max(weights.values())
            else:
                total_y = total + (max(weights.values()) if year else 0)
            recall = matched / total_y
            precision = matched / (sum(self.fact_idf.get(t, 1.0) for t in f.tokens) or 1.0)
            scored.append((recall, precision, f))
        if not scored:
            return [], 0.0
        best = max(s[0] for s in scored)
        if best < MIN_RECALL:
            return [], best
        rows = [(p, f) for r, p, f in scored if r >= best - 1e-9]

        # Qualifiers: refinance only if asked; model year if given; tier containing the amount.
        if "refinanc" not in text and "refi" not in text:
            kept = [(p, f) for p, f in rows if "refinanc" not in f.label.lower() and "refinanc" not in f.topic.lower()]
            rows = kept or rows
        if year:
            kept = [(p, f) for p, f in rows if f.years and f.years[0] <= year <= f.years[1]]
            rows = kept or rows
        if amounts:
            a = amounts[0]
            kept = [(p, f) for p, f in rows if f.tier and f.tier[0] <= a <= f.tier[1]]
            if kept:
                caps = [f.tier[1] for _, f in kept if f.tier[0] == 0.0 and math.isfinite(f.tier[1])]
                if caps and len(caps) == len(kept):
                    kept = [(p, f) for p, f in kept if f.tier[1] == min(caps)]
                rows = kept
        # One page only: official rate sheet first, then newest fetch.
        rows.sort(key=lambda pf: (-pf[1].priority, pf[1].fetched_at and -ord(pf[1].fetched_at[0]), -pf[0]))
        top = max(f.priority for _, f in rows)
        rows = [(p, f) for p, f in rows if f.priority == top]
        rows = [f for _, f in rows]
        return rows[:MAX_ROWS], best

    # ---- paragraph fallback -----------------------------------------------
    def best_paragraph(self, question):
        """BM25 best paragraph with a keyword-coverage guard; None when nothing fits."""
        q = list(dict.fromkeys(tokenize(question)))
        if not q:
            return None
        covered_total = sum(self.para_idf.get(t, 0.0) for t in q) or 1.0
        best = None
        for url, para, counts in self.paragraphs:
            length = sum(counts.values())
            score = cover = 0.0
            for t in q:
                tf = counts.get(t, 0)
                if tf:
                    idf = self.para_idf.get(t, 0.0)
                    score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / self.avg_len))
                    cover += idf
            cover /= covered_total
            if best is None or score > best[0]:
                best = (score, cover, url, para)
        if not best or best[0] < 2.0 or best[1] < 0.6:
            return None
        return best[2], best[3]


def fmt_pct(v):
    """Format a percentage with two decimals."""
    return f"{v:.2f}%"


def render(fact):
    """One-line member-facing rendering of a fact."""
    v = fact.values
    parts = []
    if "rate" in v:
        parts.append(f"rate {fmt_pct(v['rate'])}")
    if "apy" in v:
        parts.append(f"APY {fmt_pct(v['apy'])}")
    if "apr" in v:
        parts.append(f"APR {fmt_pct(v['apr'])}")
    if "min_earn" in v and "apr" not in v and len(parts) < 3 and False:
        parts.append(f"min ${v['min_earn']:,.0f}")
    name = fact.label or fact.topic
    if fact.topic and fact.topic.lower() not in name.lower():
        name = f"{fact.topic}: {name}"
    return f"{name}: " + ", ".join(parts)


class FactStore:
    """Structured fact store satisfying the KnowledgeSystem contract."""

    def __init__(self):
        self.stores = []  # SnapshotStore, oldest first

    def ingest(self, snapshot: Snapshot) -> None:
        """Parse the snapshot's tables into facts; called oldest first."""
        self.stores.append(SnapshotStore(snapshot))

    def _store_for(self, as_of):
        """Latest store taken on or before `as_of` (newest when None)."""
        if not self.stores:
            return None
        if not as_of:
            return self.stores[-1]
        eligible = [s for s in self.stores if s.taken_on <= as_of]
        return eligible[-1] if eligible else None

    def answer(self, question, as_of=None):
        """Look the question up in the fact table; fall back to paragraph search; else say 'I don't know'."""
        store = self._store_for(as_of)
        if store is None:
            return Answer(text="I don't know.", confident=False)
        rate_intent = bool(RATE_INTENT & set(tokenize(question, keep_stop=True)))
        facts, _ = store.lookup(question)
        if facts:
            facts = self._apply_superlative(question, facts)
            text = "\n".join(render(f) for f in facts)
            note = None
            alt = self._alternative(store, facts)
            if alt:
                note = alt
            return Answer(
                text=text,
                sources=[Source(url=facts[0].url, snapshot_id=store.snapshot_id, quote=text[:200])],
                conflict_note=note,
            )
        if rate_intent:
            return Answer(
                text="I don't know. The website doesn't publish that rate; please contact Dupaco directly.",
                confident=False,
            )
        hit = store.best_paragraph(question)
        if not hit:
            return Answer(text="I don't know.", confident=False)
        url, para = hit
        return Answer(text=para[:600], sources=[Source(url=url, snapshot_id=store.snapshot_id, quote=para[:200])])

    @staticmethod
    def _apply_superlative(question, facts):
        """'highest'/'lowest' questions return only the extreme row."""
        words = set(tokenize(question))
        for word, pick in (("highest", max), ("best", max), ("maximum", max), ("lowest", min), ("minimum", min)):
            if word in words:
                metric = next((m for m in ("apy", "apr", "rate") if all(m in f.values for f in facts)), None)
                if metric:
                    return [pick(facts, key=lambda f: f.values[metric])]
        return facts

    @staticmethod
    def _alternative(store, facts):
        """Mention a differing value for the same row on another page, if any (conflict note)."""
        for f in facts[:1]:
            for g in store.facts:
                if g.url != f.url and g.label == f.label and g.topic == f.topic and g.values != f.values:
                    return f"{g.url} shows {g.values}; using {f.url} (official rate sheet / newer)."
        return None

    def conflicts(self):
        """Same row + metric, different values on different pages of the newest snapshot."""
        if not self.stores:
            return []
        store = self.stores[-1]
        groups = defaultdict(list)
        for f in store.facts:
            key = (" ".join(sorted(f.tokens)), f.tier, f.years)
            groups[key].append(f)
        out = []
        for facts in groups.values():
            by_url = {}
            for f in facts:
                by_url.setdefault(f.url, f)
            if len(by_url) < 2:
                continue
            for metric in ("rate", "apy", "apr"):
                have = [(u, f) for u, f in by_url.items() if metric in f.values]
                if len({f.values[metric] for _, f in have}) < 2:
                    continue
                have.sort(key=lambda uf: (-uf[1].priority, uf[1].fetched_at), reverse=False)
                win = max(have, key=lambda uf: (uf[1].priority, uf[1].fetched_at))[1]
                out.append(
                    Conflict(
                        topic=f"{win.topic} {win.label} {metric.upper()}",
                        values=[fmt_pct(f.values[metric]) for _, f in have],
                        sources=[Source(url=u, snapshot_id=store.snapshot_id) for u, _ in have],
                        resolution=fmt_pct(win.values[metric]),
                        reason="official rate sheet" if win.priority > 1 else "newest page",
                    )
                )
        return out


def build():
    """Create the knowledge system used by the harness."""
    return FactStore()
