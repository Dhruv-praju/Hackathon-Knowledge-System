# System design: knowledge system that knows what's true *now*

Status as of 2026-11-10. Sections marked **Built** exist in this repo; sections marked **Planned** do not.

## 1. Problem

Answer credit-union members' questions from five crawls of the Dupaco website (2026-05-12 to 2026-09-28, ~640-705 pages each). The crawls disagree with each other:

- Rates change between crawls, and old pages stay up.
- Expired promotions and old press releases still show old numbers.
- Tables hold tiers (balance, term, loan amount, model year) that look like conflicts if read carelessly.
- Rate vs APY vs APR, and personal vs business products, are different facts.
- Some answers are not on the site at all, and the system must say so.

### Contract (`harness/contract.py`)

| Method | Meaning |
|---|---|
| `ingest(snapshot)` | Called once per snapshot, oldest first. Untimed, must finish within 30 minutes in total. |
| `answer(question, as_of=None)` | Newest snapshot if `as_of` is None, otherwise the site as it stood on that date. Returns text, sources, `confident`, optional `conflict_note`. |
| `conflicts()` | Every contradiction, the values, the sources, the winner and why. |

### Scoring

| Pillar | Weight | Notes |
|---|---|---|
| Accuracy | 55% | Current, correctly tiered value. A stale value scores zero even if the right one is also present. "I don't know" is correct when the site is silent. |
| Conflicts | 15% | Real contradictions listed, right winner picked. |
| Consistency | 15% | Reworded questions and repeats give the same answer. Multiplied by accuracy. |
| Speed | 15% | Full marks at 2 s or less per answer, zero at 20 s. Multiplied by accuracy. |

## 2. Goals and non-goals

**Goals**
- Return the current, correctly tiered value with a source.
- Refuse honestly when the site is silent.
- Give identical answers to repeated and reworded questions.
- Answer in under 2 s.
- List contradictions with a justified winner.

**Non-goals**
- Live crawling.
- Multi-turn dialogue or personal account data.
- Any answer not grounded in the snapshots.

## 3. Data

Each snapshot is a full copy of the site: `manifest.json` (url, title, `fetched_at`, file) plus `pages/*.md`. Facts about the data that drive the design:

- About 3,300 pages in total, 2,826 unique by content.
- The official rate sheet (`/rates`, `/business-rates`) is a clean set of markdown tables, one section per product.
- Prose pages hold branch hours, addresses, closures, eligibility and program details.
- Blogs, calculators and resources pages are large, repetitive, and often stale or off-topic.

## 4. Target architecture

Three layers, cheapest and most exact first. Each layer either answers or hands the question down.

```
                 question (+ as_of)
                        |
              [0] snapshot selector
                        |
        +---------------+----------------+
        |                                |
 [1] table fact lookup           [2] LLM fact lookup
 (exact, rule-based)             (prose facts, extracted at ingest)
        |                                |
        +----------- miss ---------------+
                        |
              [3] RAG + LLM judge  (hard / uncovered questions)
                        |
              [4] refusal / "I don't know"
```

### 4.1 Snapshot selector (Built)
- `as_of=None` picks the newest snapshot. A date picks the newest snapshot taken on or before it. A date before the first snapshot returns "I don't know".
- Facts and indexes are kept per snapshot, so "as of" questions and "what changed" questions need no extra machinery.

### 4.2 Layer 1: table facts (Built: `facts/`, copied into `llm_facts/` and `rag_judge/`)

**Ingest.** Every markdown table row becomes a `Fact`:

| Field | Example |
|---|---|
| topic / section | "Regular Certificates" / "Certificates" |
| label | `2024-27 Model Year \| 60 Months` |
| values | `{rate, apy, apr, min_open, min_earn}` |
| tier | (lo, hi) parsed from `$10,000 – $24,999.99`, `$25,000+`, `$100k or less` |
| model-year range | `2024-27`, `2016 (and older)` |
| source | url, snapshot id, `fetched_at`, priority (official rate sheet = 3, other pages = 1) |

**Lookup.**
1. Tokenize the question (drop stopwords and generic words such as "rate", "current", "balance"; stem plurals). Pull out dollar amounts and a year.
2. Score each row by the idf-weighted share of question words it explains (recall). Words that appear in no table count as strong mismatches; synonyms map "mortgage" to home/purchase, "CD" to certificate.
3. Require recall of at least 0.6, otherwise no match.
4. Apply qualifiers: refinance rows only if asked; model year must lie in the row's range; the amount must fall inside the row's tier (for "or less" tiers, the smallest cap that fits).
5. Keep only the highest-priority source (official rate sheet over blogs).
6. "Highest" and "lowest" return a single extreme row.
7. A rate question with no matching row returns a non-confident "the website doesn't publish that rate".

### 4.3 Layer 2: LLM fact extraction (Built: `llm_facts/`)

**Ingest.** For each prose page (about 900 unique after skipping blog, calculator, resources, privacy and tag pages), an LLM returns structured facts through a forced tool call:

`{entity, attribute, value, qualifiers, effective, status (current|expired), quote}`

- Cached on disk by SHA-256 of prompt version, model, url and page text. A page unchanged across snapshots is extracted once; re-runs make no API calls.
- 8 parallel workers, 1,500 s budget, then cached pages only. A failed call skips the page, it never aborts ingest.
- Measured: 901 calls, 610 s, about 3,000 facts per snapshot.

**Lookup.** idf-weighted recall of question words against each fact's entity, attribute and qualifiers (threshold 0.6). Expired facts are dropped, duplicates collapse to the most recently fetched.

### 4.4 Layer 3: RAG + LLM judge (Built: `rag_judge/`, not yet measured)

**Ingest.** Tables as in 4.2, plus sliding windows of about 900 characters over consecutive paragraphs, de-duplicated across pages, with URL tokens added and calculator/blog pages down-weighted (x0.6).

**Answer.** One LLM call per question with:
- the exact table rows (labelled as the official rate sheet),
- the best BM25 windows from the top 8 pages, each tagged with url and crawl date.

The system prompt carries the authority rules (section 5). The model answers through a forced tool call: `{answer, confident, source_urls, conflict_note}`. Cited URLs are restricted to those actually shown. Answers are cached by (snapshot, question). Any LLM error falls back to the table-only answer.

### 4.5 Refusal (Built)
Order of fallbacks inside each solution: table rows, then LLM facts, then rate-intent refusal, then paragraph BM25 with coverage thresholds, then "I don't know". A non-confident answer carries `confident=False`.

### 4.6 Recommended combination (Planned)

Run the layers in order and let the judge arbitrate only when needed:

1. Table lookup hit: answer from it, no LLM.
2. LLM-fact hit with strong match: answer from it, no LLM.
3. Otherwise run Layer 3, passing both the table rows and the matched LLM facts as extra context.

This keeps the fast, exact path for the common rate questions (the 2 s speed target, deterministic answers) and spends an LLM call only on prose questions.

## 5. Authority rules (the "what is true now" policy)

These are encoded in code (Layers 1 and 2) and in the judge prompt (Layer 3).

1. **Newest wins.** For the same entity and attribute, prefer the latest crawl date within the chosen snapshot.
2. **Official over informal.** Rate sheet and product-page rate tables beat blogs, newsletters and press releases.
3. **Expired is not current.** Drop promotions, events and offers the text says have ended.
4. **Qualifiers are part of the key.** Product, term, tier, loan amount, model year, purchase vs refinance, rate vs APY vs APR, personal vs business. Different tiers are different facts.
5. **Tiers by amount.** With an amount or term, return the matching tier; without one, lead with the headline tier and mention others.
6. **Honesty.** If the site does not state it, say so and refer the member to Dupaco. Never invent a number, date, address or URL.
7. **Single answer.** Do not list every historical value to hedge.

## 6. Conflict detection (`conflicts()`)

**Built (table level).** On the newest snapshot, group rows by their token set, tier and year range. If two pages give different values for the same metric (rate, APY or APR), emit a `Conflict` with the values, sources, the winner and the reason ("official rate sheet" or "newest page").

**Built (LLM level, `llm_facts`).** Group current LLM facts by entity, attribute and qualifiers; different values become a conflict, winner is the most recently fetched.

**Planned.**
- Cross-snapshot changes (a value that moved between crawls) reported as updates with the old value marked stale.
- Prose-vs-table disagreements (e.g. a blog quoting an old rate).
- Validating the output against the real contradictions. It has not been checked yet, and `conflict_resolution` scores 0.00 for Approaches 2 and 3.

## 7. Consistency and speed

| Concern | Mechanism |
|---|---|
| Same question twice | Layers 1 and 2 are deterministic. Layer 3 caches answers by (snapshot, question). |
| Reworded questions | Question words are stemmed, stopwords, filler and month names are ignored, and synonyms are mapped. Layer 3 can still differ between wordings. Planned: normalise the question to a canonical key before caching. |
| Speed | Layers 1 and 2 answer in about 10-20 ms. Layer 3 depends on LLM latency (a 12 s timeout, then fallback). |
| Ingest budget | Table parse about 1.5 s. LLM extraction is capped at 1,500 s and resumable from cache. |

## 8. Repository layout

```
starter/       original BM25 / hybrid (BM25 + fastembed) baseline
facts/         Approach 2: table fact store (stdlib only)
llm_facts/     Approach 3: table facts + LLM-extracted prose facts (cache in llm_facts/cache/)
rag_judge/     Approach 4: table rows + BM25 windows + LLM judge (cache in rag_judge/cache/)
ask.py         ask one question, or interactive loop with a single ingest
harness/       scorer, contract
RESULTS.md     measured scores for every approach
```

Run any approach with `python3 harness/run_eval.py --submission <folder> --eval eval/public_eval.jsonl`. `llm_facts` and `rag_judge` need `pip install anthropic` and `ANTHROPIC_API_KEY`; without a key both fall back to the table solution.

## 9. Measured results (public eval, 40 questions)

| Approach | Score |
|---|---|
| BM25 only | 0.2492 |
| Hybrid BM25 + embeddings | 0.3120 |
| Table fact store | 0.4876 |
| Table + LLM-extracted facts | 0.5527 |
| RAG + LLM judge | not yet measured (first run failed on a `temperature` argument, now fixed) |

Details by category are in `RESULTS.md`.

## 10. Known gaps and risks

| Gap | Effect | Planned fix |
|---|---|---|
| LLM-fact lookup matches entity and attribute only, not value or quote | Galena address, branch count (22) and Hy-Vee still-open return "I don't know" | Match against value and quote, rank by overlap |
| Wrong fact wins on loose word overlap | Forbes returns another award, locator URL returns an ATM count, "join online" returns the wrong URL | Rank by overlap, prefer URL-valued facts for link questions |
| Business card table not parsed | Business cards missing from card-list answers | Fix the header classification for 2-column tables |
| Link questions refused as rate questions | `dup-008` fails | Detect link/URL intent before the rate-refusal rule |
| BM25 retrieval misses the right page | The judge cannot answer (Galena address) | Add embedding retrieval and fuse with BM25; always add LLM-fact hits to the judge context |
| Layer 3 wording sensitivity | Possible consistency loss | Canonicalise the question before caching, constrain the output format |
| `conflicts()` unvalidated | Conflict score unknown | Compare against known contradictions in the data |
| API dependency | Needs a key at ingest or answer time, public site text leaves the machine | Caches make re-runs offline; table-only fallback when there is no key |

## 11. Roadmap

1. Run `rag_judge` and record its score in `RESULTS.md`.
2. Fix LLM-fact lookup (value and quote matching, ranking, link intent).
3. Parse the business card table.
4. Build the combined pipeline (4.6) as a single submission folder.
5. Validate and extend `conflicts()` (cross-snapshot changes, prose-vs-table).
6. Write `NOTES.md` (required by the hand-in) listing the API key and packages needed.
