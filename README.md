# LLM Data Cleaning — master's thesis experiment harness

Compares 6 LLMs chosen for **type diversity and cost** (reasoners, flash/thinking,
cheap chat, open instruct, cheap closed generalist) plus 2 non-LLM baselines
(rapidfuzz, sentence-transformers) on `accept_rename`, `accept_no_rename`,
`decline`, `suggest`. See `llm_cleaner/experiments/_models.py` for IDs.

## Repository layout

```
.
├── docker-compose.yml          Postgres + ETL container
├── Dockerfile                  Python image used by docker-compose
├── requirements.txt
├── .env.example                Copy to .env and add OPENROUTER_API_KEY
├── db/init/01_schema.sql       Postgres schema (test_cases + runs)
├── harness.py                  CLI: seed / run / analyze / list
├── script.py                   Backwards-compatible shim (legacy folder mode)
├── llm_cleaner/
│   ├── prompts.py              P0..P4 + P_with_confidence prompt builders
│   ├── clients/                OpenRouter + fuzzy + embedding cleaners
│   ├── core.py                 RunContext + run_matrix + run_batch_matrix
│   ├── etl.py                  Postgres connection + insert + failure-mode classifier
│   ├── seed.py                 Walks CSV folders into test_cases rows
│   ├── analyze.py              Metrics + plots
│   └── experiments/            One file per experiment (exp01..exp09)
├── not_mixed_tests/            Original test cases (with expected.csv)
├── mixed_tests/                Original mixed tests (with expected.csv)
├── tests_data/                 Curated extras
│   ├── extra/                    typo_ladder, abbreviations, synonyms, ambiguous,
│   │                             noise, edge_decline, edge_suggest, long_inputs
│   ├── adversarial/              prompt_injection, homoglyphs, empty_and_emoji
│   └── scope_scaling/            countries_5, countries_20, countries_50, countries_200
├── large_tests/                ~6k-row generated suite (expected.jsonl gold)
└── generate_data.py            Reproducible generator for large_tests/ (seed=42)
```

## Quick start

1. **Create `.env`** from `.env.example` and add your OpenRouter API key.
2. **Start Postgres**: `docker compose up -d postgres` (the schema file in
   `db/init/` is applied automatically on first boot).
3. **Install Python deps** locally (or use the `etl` container):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. **Seed the test cases**: `python harness.py seed`
5. **Zero-cost e2e preflight** (mock LLM, real Postgres — run this before paid jobs):
   ```powershell
   python harness.py e2e
   # or: python tools/e2e_preflight.py
   ```
   Asserts every experiment writes `runs` rows (models/prompts/modes/confidence)
   and that `analyze` works. Spends **$0** on OpenRouter (`LLM_MOCK=1`).
6. **Optional live smoke** (tiny real API spend): `python harness.py run --experiment exp09_smoke`
7. **Analyze**:
   `python harness.py analyze --experiment-id <uuid> --out reports/`

## Running an experiment

```powershell
python harness.py list                                                # see all experiments
python harness.py run --experiment exp01_core_matrix                  # RQ1 headline
python harness.py run --experiment exp02_prompt_ablation --limit 10   # dev mode
python harness.py run --experiment exp03_batch_vs_single
python harness.py run --experiment exp04_variance                     # 10 reps, RQ5
python harness.py run --experiment exp05_adversarial
python harness.py run --experiment exp06_scope_scaling
python harness.py run --experiment exp07_baselines                    # local, no API cost
python harness.py run --experiment exp08_calibration
```

Each invocation prints an `experiment_id` UUID at the end; pass it to
`analyze` to produce CSVs and PNGs under `reports/<experiment_id>/`.

The `--limit N` flag truncates the test set to N for development runs
and applies to every experiment.

### Suites: small vs large

By default experiments run on the **small** suite only
(`not_mixed_tests/`, `mixed_tests/`, `tests_data/`). The generated
`large_tests/` corpus (~6188 inputs across 10 domains + a big mixed set)
is seeded into Postgres but excluded unless you opt in:

```powershell
python harness.py run --experiment exp07_baselines --suite large --limit 50
python harness.py run --experiment exp01_core_matrix --suite all --limit 20
```

`--suite` values: `small` (default), `large`, `all`.
On `large`/`all`, `exp01_core_matrix` defaults to **1 repetition** (use
`--repetitions 3` only if budget allows).

Gold labels in `large_tests/` use predicate `expected.jsonl` (referent-based
`one_of` / `empty` / `same_referent_as_input`). The seeder maps the **primary**
allowed entry into `expected_state` / `expected_value` for the harness. Rebuild
the corpus with `python generate_data.py` (fixed seed).

## Experiment registry

| Name | RQ | Models | Prompts | Modes | Reps | Notes |
|------|----|--------|---------|-------|------|-------|
| `exp01_core_matrix` | RQ1 | 6 | P0 | single | 3 (1 on large) | Headline accuracy; 1 rep on `--suite large` |
| `exp02_prompt_ablation` | RQ2 | 6 | P0..P4 | single | 3 | Is prompt > model? |
| `exp03_batch_vs_single` | RQ3a | 6 | P0 | single + batch x4 shuffles | 3 | Order bias |
| `exp04_variance` | RQ5 | 3 cheap | P0 | single | 10 | T ∈ {0, 0.1, 0.7, 1.0} |
| `exp05_adversarial` | RQ1 robust | 6 | P0 | single | 3 | Injection, homoglyphs, emoji |
| `exp06_scope_scaling` | RQ3b | 3 cheap | P0 | single | 3 | Scope sizes 5/20/50/200 |
| `exp07_baselines` | RQ4 | rapidfuzz + embed | n/a | single | 1 | Free, deterministic |
| `exp08_calibration` | RQ5 | 6 | P_with_confidence | single | 3 | Reliability diagram, ECE |
| `exp09_smoke` | sanity | 1 cheap | P0 | single | 1 | End-to-end check |

To **swap which experiment runs you only change the `--experiment` flag** —
no code edits, no refactors. To override the model list without editing
code, set `LLM_MODELS="openai/gpt-oss-120b,openai/gpt-4o-mini"`.

### Models (default)

| OpenRouter ID | Role |
| --- | --- |
| `openai/gpt-oss-120b` | Open reasoner (very cheap) |
| `deepseek/deepseek-r1-0528` | R1-class open reasoner |
| `google/gemini-2.5-flash-lite` | Flash + reasoning + MM-capable |
| `openai/gpt-4o-mini` | Classic cheap chat (no reasoning API) |
| `meta-llama/llama-4-scout` | Open instruct + MM-capable |
| `google/gemma-4-31b-it:free` | Free reasoner + MM-capable (rate limits) |

### Scoring (analyze)

`python harness.py analyze ...` re-scores runs against on-disk gold
(`expected.jsonl` preferred). For `same_referent_as_input` (suggest) it uses
a deterministic cascade: normalized equality → aliases → Levenshtein ≥ 0.85.
Reports **lenient** accuracy (any `allowed[]` entry; primary heatmaps) and
**strict** accuracy (first / preferred entry), plus `validation_tiers.csv`.
Acceptance with a value not in scope is `hallucinated_scope`; treating a
scope surface-variant as suggest is `wrong_state`.

## Adding a new experiment

1. Create `llm_cleaner/experiments/expNN_<name>.py` with a top-level
   `def run(ctx: RunContext) -> None:` that calls `run_matrix` or
   `run_batch_matrix`.
2. Add the name to `EXPERIMENT_NAMES` in `llm_cleaner/experiments/__init__.py`.

That's it.

## Adding a new test case

Create a folder under one of the data roots with:

```
tests_data/extra/<group>/<case>/
    scope.csv          # one column, header optional
    input_data.csv     # column 'raw_value'
    expected.csv       # OR expected.jsonl (see large_tests/)
                       # csv columns: expected_state, expected_value, notes
                       # one row / line per input_data row
```

Then re-run `python harness.py seed` (it upserts, so duplicates are safe).

If the parent folder name is not in `SCENARIO_FROM_FOLDER` in
`llm_cleaner/seed.py`, add it.

## Database schema

`test_cases` holds one row per (folder, raw_value); `runs` holds one row
per LLM call. See `db/init/01_schema.sql`. The view `runs_summary`
gives quick per-experiment aggregates:

```sql
SELECT * FROM runs_summary WHERE experiment_name = 'exp01_core_matrix';
```

## OpenRouter feature matrix

Per the plan §9, everything goes through the standard
`POST /api/v1/chat/completions` endpoint already used in the original
script. The harness:
- attempts `response_format: json_schema` for P3, falls back to
  `json_object`, falls back to text+parse;
- captures `usage.cost`, `usage.prompt_tokens`, `usage.completion_tokens`;
- captures the actual `provider` (which physical host served the call)
  for the threats-to-validity section;
- passes `seed` through for deterministic-mode comparisons in
  `exp04_variance`.

Each call records which `response_format_used` it ended up with in the
`runs` table so model-by-model schema support can be reported.

## Cost ballpark (hard cap ~$100)

Prices from live OpenRouter `/api/v1/models` (prompt/completion $/1M),
assuming ~900 input tokens and ~150 output (chat) / ~700 (reasoners) per call.
`deepseek-r1-0528` dominates cost (~$0.002/call); gemma free is $0.

| Plan | Calls (approx) | Blended est. | Worst (all as R1) |
|------|----------------|--------------|-------------------|
| Small suite, all exp01–09 (defaults) | ~6.7k | **~$3–4** | ~$14 |
| Large `exp01`, 6 models × **1 rep** (default on `--suite large`) | ~37k | **~$19** | ~$78 |
| Large `exp01`, 6 models × 3 reps (opt-in `--repetitions 3`) | ~111k | ~$57 | ~$235 |

**Recommended thesis budget under $100:**

```powershell
# 1) All RQs on curated small suite (~$4)
python harness.py run --experiment exp01_core_matrix
python harness.py run --experiment exp02_prompt_ablation
# ... exp03..exp08, exp09_smoke

# 2) Large confirmation pass — 1 rep is the default for suite=large
python harness.py run --experiment exp01_core_matrix --suite large
```

Expected total **~$22–35** (with token/retry headroom still under **$100**).
Do **not** pass `--repetitions 3` on the full large suite unless you have
budget left — that is the ~$235 worst-case path. Use `--limit N` for
dev; `--repetitions N` overrides the suite default.

## Legacy mode

The original CLI still works:

```powershell
python script.py not_mixed_tests/accept_with_renaming/2A
```

It uses the new OpenRouter client under the hood but doesn't touch the
database; useful for quick smoke checks on individual folders.
