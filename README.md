# Hackathon: Build a Knowledge System That Knows What's True *Now*

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
| Final score | **0.8748** |
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
| stale_trap | 0.98 |
| list_completeness | 0.75 |
| entity_attribution | 1.00 |
| current_value | 1.00 |
| should_refuse | 1.00 |

The `conflicts()` score is not part of this output because the public run has no `--conflicts-gold` file. Known misses: certificate read-back (needs `--judge`), and three questions that lose the 15% source credit because the expected page (home page, `/contact`, savings-account) is not where our facts came from.
