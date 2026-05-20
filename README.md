# LLM Data Cleaning — master's thesis experiment harness

Compares 6 LLMs (GPT-5.5, DeepSeek V4 Pro, Claude Sonnet 4.6, Gemini 3.1 Pro,
Llama 3.3 70B, GPT-4o-mini) and 2 non-LLM baselines (rapidfuzz, sentence-transformers)
on four data-cleaning scenarios: `accept_rename`, `accept_no_rename`,
`decline`, `suggest`. See `llm_data_cleaning_paper_*.plan.md` for the full
design.

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
├── not_mixed_tests/            Original test cases (with new expected.csv files)
├── mixed_tests/                Original mixed tests (with new expected.csv files)
└── tests_data/                 New test cases
    ├── extra/                    typo_ladder, abbreviations, synonyms, ambiguous,
    │                             noise, edge_decline, edge_suggest, long_inputs
    ├── adversarial/              prompt_injection, homoglyphs, empty_and_emoji
    └── scope_scaling/            countries_5, countries_20, countries_50, countries_200
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
5. **Smoke test the pipeline**: `python harness.py run --experiment exp09_smoke`
   (1 model x 3 cases x 1 rep, costs roughly $0.001)
6. **Analyze the smoke run**:
   `python harness.py analyze --experiment-id <uuid printed in step 5> --out reports/`

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

## Experiment registry

| Name | RQ | Models | Prompts | Modes | Reps | Notes |
|------|----|--------|---------|-------|------|-------|
| `exp01_core_matrix` | RQ1 | 6 | P0 | single | 3 | Headline accuracy numbers |
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
code, set `LLM_MODELS="openai/gpt-5.5,anthropic/claude-sonnet-4.6"`.

## Adding a new experiment

1. Create `llm_cleaner/experiments/expNN_<name>.py` with a top-level
   `def run(ctx: RunContext) -> None:` that calls `run_matrix` or
   `run_batch_matrix`.
2. Add the name to `EXPERIMENT_NAMES` in `llm_cleaner/experiments/__init__.py`.

That's it.

## Adding a new test case

Create a folder under one of the data roots with two CSVs:

```
tests_data/extra/<group>/<case>/
    scope.csv          # one column, header optional
    input_data.csv     # column 'raw_value'
    expected.csv       # columns: expected_state, expected_value, notes
                       # one row per input_data row
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

## Cost ballpark

See plan §8. Running all 8 substantive experiments on the full test set
should cost roughly **$200–$400** at 2026 OpenRouter prices, comfortably
inside the $1000 budget. `--limit 5` brings any individual experiment to
under $5 for development.

## Legacy mode

The original CLI still works:

```powershell
python script.py not_mixed_tests/accept_with_renaming/2A
```

It uses the new OpenRouter client under the hood but doesn't touch the
database; useful for quick smoke checks on individual folders.
