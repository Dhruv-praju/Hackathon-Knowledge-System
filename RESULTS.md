# Eval results

Public eval: `eval/public_eval.jsonl` (40 questions). Scores are `final_score` from `harness/run_eval.py`.
Consistency and speed were 1.0 in every run, and they are multiplied by accuracy, so `final_score` equals accuracy.

Last updated: 2026-11-10.

## Summary

| # | Approach | Folder / branch | Final score | Stale-trap hits | Ingest |
|---|---|---|---|---|---|
| 1a | BM25 keyword search | `starter` on `feature/keyword_search` | 0.2492 | 1 | 0 s |
| 1b | Hybrid: BM25 + embeddings (fastembed, RRF) | `starter` on `feature/semantic` | 0.3120 | 0 | 767 s |
| 2 | Structured fact store (table parsing + rules) | `facts/` | 0.4876 | 0 | 1.4 s |
| 3 | LLM extraction at ingest + table facts | `llm_facts/` | **0.5527** | 0 | 610 s (first run, 901 API calls; cached after) |

## Score by category

| Category (n) | 1a BM25 | 1b Hybrid | 2 Facts | 3 LLM facts |
|---|---|---|---|---|
| branch_info (6) | 0.17 | 0.17 | 0.17 | 0.64 |
| conflict_resolution (3) | 0.33 | 0.33 | 0.00 | 0.00 |
| current_value (3) | 0.00 | 0.00 | 0.33 | 0.67 |
| entity_attribution (4) | 0.46 | 0.46 | 0.71 | 0.71 |
| list_completeness (4) | 0.00 | 0.00 | 0.36 | 0.36 |
| should_refuse (3) | 0.33 | 0.33 | 0.67 | 0.33 |
| stale_trap (7) | 0.31 | 0.26 | 0.65 | 0.36 |
| temporal_as_of (4) | 0.50 | 0.50 | 0.25 | 0.75 |
| tier_disambiguation (6) | 0.17 | 0.67 | 1.00 | 1.00 |

## Commands

```bash
# 1a / 1b (1b needs: pip install -r starter/requirements.txt)
python3 harness/run_eval.py --submission starter --eval eval/public_eval.jsonl

# 2: no installs needed (standard library only)
python3 harness/run_eval.py --submission facts --eval eval/public_eval.jsonl

# 3: needs `pip install -r llm_facts/requirements.txt` and ANTHROPIC_API_KEY
python3 harness/run_eval.py --submission llm_facts --eval eval/public_eval.jsonl
```

Approach 3 caches every page extraction in `llm_facts/cache/`. Re-runs reuse it, so ingest is fast and makes no API calls.
Without an API key it falls back to approach 2 and scores 0.4876.

## What changed between runs

- **1b over 1a:** embeddings fixed the stale-trap hit and raised tier questions from 0.17 to 0.67, but ingest grew to 13 minutes.
- **2 over 1b:** parsing tables into rows (term, tier, model-year and amount aware) fixed the tier questions and most rate lookups, and made ingest instant. Rate questions with no matching row now refuse with "not published".
- **2, tuning:** treating words that appear in no table as strong evidence against a match fixed the 7/1 ARM and construction-loan answers. Filler words ("highest", "their", "was", month names) then had to be ignored to stop good matches being rejected. The score ended where it began (0.4876) but the refusals are correct.
- **3 over 2:** LLM-extracted facts answer prose questions (branch hours, Hy-Vee closing, contact details), lifting `branch_info`, `current_value` and `temporal_as_of`.

## Known failures (approach 3, latest run)

- **Wrong fact returned:** Forbes returns an "Employ Humanity" award, the branch locator URL returns an ATM count, and "join online" returns `dupaco.com/about/join` instead of `join.dupaco.com`.
- **Fact extracted but not found:** Galena address, branch count (22) and Hy-Vee-still-open return "I don't know". Lookup matches only entity, attribute and qualifiers, not value or quote text.
- **Prose answers:** membership eligibility, contact update, mobile wallet and "same lobby hours" return a heading or a loose paragraph.
- **Credit cards:** the business card table is not parsed, so business cards are missing from `dup-034` and `dup-040`.
- **Link questions:** `dup-008` (CD rates link) is refused as a rate question.
- **Conflicts:** `conflict_resolution` is 0.00 for approaches 2 and 3, and the `conflicts()` output has not been checked against the real data.

## Next steps

1. Match question words against each LLM fact's value and quote, and rank by overlap so the best fact wins.
2. Handle "link" and "URL" questions by matching facts whose value is a URL.
3. Parse the business card table.
4. Return the top two or three paragraphs when no fact matches well.
5. Review `conflicts()` output.
