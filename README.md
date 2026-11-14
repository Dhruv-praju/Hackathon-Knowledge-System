# Hackathon: Build a Knowledge System That Knows What's True *Now*

## The problem in one minute

Our assistant answers credit-union members' questions ("What's the 6-month certificate rate?", "How much is a lost debit card?") using the credit union's own website.

We download (crawl) that website again and again over time. Each download is a **snapshot**. Websites are messy:

- A rate changes, but an old page or PDF still shows the old number.
- A promotion ends, but its page is never taken down.
- One page lists the **rate**, another the **APY**, and they look like they disagree.
- A table has tiers (by balance, by term). Read it carelessly and every tier looks like a conflict.
- Old newsletters and press releases describe offers from years ago.

When we combined these snapshots, our system **gave members wrong or out-of-date answers**. Every eval question in this kit comes from a real problem like that.

**Your mission:** build a system that reads the snapshots in date order and answers questions the way a careful human expert would: with the **current** value, the **right** tier, a **source**, and an honest "I don't know" when the site doesn't say.

## What's in the box

```
snapshots_zip/           the five website downloads, one zip per date (oldest first)
snapshots/README.md      what each download is and what changed between them
eval/public_eval.jsonl   40 practice questions with expected answers
starter/solution.py      a working (but naive) solution to copy from
harness/                 the scorer we will use (same one you can run)
```

**You don't need to unzip anything.** The first time you run the scorer (below), it unpacks the zips into `snapshots/`, with one folder per download, each holding a `manifest.json` (url, title, date) and the pages as `pages/*.md`. After that the folders are there to browse. If you'd rather look first, unzip them yourself into `snapshots/`; the scorer leaves existing folders alone.

## What you must hand in

A folder with a file called `solution.py`, containing a `build()` function that returns an object with **three methods**:

| Method | What it does |
|---|---|
| `ingest(snapshot)` | We call it once per snapshot, oldest first. Learn from it. |
| `answer(question, as_of=None)` | Answer a member's question. Without `as_of`, answer from the newest snapshot. With a date, answer as the website stood on that day. |
| `conflicts()` | List every place where the website contradicts itself, which value you believe is right, and why. |

The exact shapes are defined in `harness/contract.py`; `starter/solution.py` already follows them. **Don't rename anything.** We plug your folder straight into our scorer.

You can use any tools you like inside: search, embeddings, a graph, an LLM, plain rules. Add a `requirements.txt` if you need packages, and say in a short `NOTES.md` which API keys (if any) you need.

## How to check your work

From the top folder of this kit, run:

```bash
python harness/run_eval.py --submission starter --eval eval/public_eval.jsonl
```

The first run takes a few extra seconds while it unpacks the snapshots. Then it loads all five downloads into the starter, asks the 40 practice questions, and prints your score. To test your own solution, point `--submission` at your folder.

The starter scores low on purpose. Watch your score go up as you improve. Needs Python 3.9 or newer.

## How you will be scored

We run a **larger, hidden question set** with the same format and the same scorer. Four things count:

| | Weight | What we look for |
|---|---|---|
| **Accuracy** | 55% | The current, correctly-tiered value is in your answer. Repeating an old or wrong value scores **zero** for that question, even if the right value is also there, so hedging by listing every number doesn't pay. Citing the right page earns a little extra. When the site doesn't say, the right answer is "I don't know". |
| **Conflicts** | 15% | How many real contradictions you list in `conflicts()`, and whether you picked the right winner. |
| **Consistency** | 15% | The same question asked in different words gets the same answer, and asking twice gives the same answer. |
| **Speed** | 15% | How quickly you answer. Most answers in 2 seconds or less earns full marks; it drops to zero at 20 seconds. |

Speed and consistency are **multiplied by your accuracy**: a fast or consistent system that's wrong doesn't earn those points. Ingest (loading the snapshots) isn't timed for scoring, but must finish within 30 minutes.

Hidden questions include re-worded versions of the public ones, "as of" date questions, and questions about things that *changed* between snapshots.

## A simple way to approach it

1. **Read the snapshots yourself first.** (Run the scorer once to unpack them, or unzip `snapshots_zip/`.) Pick five topics (a rate, a fee, branch hours) and follow them across the dates. You'll quickly see the kinds of mess.
2. **Keep history, not just the latest copy.** Remember *when* you saw each fact and *where*.
3. **Decide what "current" means.** Is the newer page right? Does an official rate sheet beat a blog post? What about an expired promotion? Write your rules down.
4. **Respect qualifiers.** "6-month" vs "12-month", "rate" vs "APY", "Prime Checking" vs "Prime Business Checking". These are different facts, not conflicts.
5. **Be honest.** If the website doesn't say, don't guess.
6. **Run the public eval often.** Read the questions you miss, not just the score.

## Rules

- Use only the snapshots provided to answer questions. No live website calls.
- Each `answer` call must finish within 60 seconds.
- The data is a real credit union's public website, shared for this event only. Don't redistribute it.

## Our solution: `llm_facts/`

Structured facts at ingest, no LLM at answer time. Table rows are parsed by rules; prose pages are read once by Claude Haiku into entity / attribute / value facts (cached on disk). Answers come from table rows, then LLM facts, then BM25, else "I don't know". The design, the rules for what counts as current, and the diagram are in [`llm_facts/NOTES.md`](llm_facts/NOTES.md).

### Run it

```bash
pip install -r llm_facts/requirements.txt          # only needed to call the LLM at ingest
export ANTHROPIC_API_KEY=...                       # optional: without it the committed llm_facts/cache/ is used
python3 harness/run_eval.py --submission llm_facts --eval eval/public_eval.jsonl
```

Optional env vars: `LLM_FACTS_MODEL`, `LLM_FACTS_WORKERS`, `LLM_FACTS_BUDGET_S`, `LLM_FACTS_CACHE` (see NOTES.md).

### Results (public eval, 40 questions)

| Metric | Value |
|---|---|
| Final score | **0.8748** (starter was far lower; first llm_facts run was 0.5527) |
| Consistency (paraphrase and repeat) | 1.0 |
| Speed (p50 / max per answer) | 0.0 s / 0.03 s, score 1.0 |
| Stale-trap hits / errors | 0 / 0 |
| Ingest time | 1.6 s with the cache (limit 30 min) |

| Category | Score |
|---|---|
| tier_disambiguation | 1.00 |
| conflict_resolution | 1.00 |
| temporal_as_of | 1.00 |
| branch_info | 0.95 |
| stale_trap | 0.91 |
| list_completeness | 0.75 |
| entity_attribution | 0.75 |
| current_value | 0.67 |
| should_refuse | 0.67 |

The `conflicts()` score is not part of this output because the public run has no `--conflicts-gold` file. Known misses: certificate read-back (needs `--judge`), a "Mullan Avenue" refusal, update-contact-info, and a student-loan question.
