# llm_data_cleaning
Project will focus on using llm to clean input data based on scope.

## Run a single test case
python script.py folder/subfolder/subfolder

## Decision rules (referent-based)
The prompt applies an ordered decision procedure:
1. Same referent test: the input names the SAME entity as a scope item, differing only in surface form (typo, casing, abbreviation, noise characters, exact alias) -> acceptance (with renaming if the surface form differs).
2. Same category test: a real, distinct item of the same category, not in scope -> suggest.
3. Otherwise (garbage, other category, placeholders like N/A) -> decline.

Tie-break: when torn between renaming and suggesting, prefer suggest (renaming a distinct entity silently corrupts data; suggesting preserves information).

## Evaluation
Gold labels are PREDICATE SPECS, not enumerated literals, because for
"suggest" the space of correct values is open (a whole spectrum of correct
surface forms exists). Each test folder contains expected.jsonl, one spec
per input:

    {"raw_value": "Dev", "allowed": [
      {"state": "acceptance", "value": {"mode": "one_of", "items": ["Software Engineer"]}},
      {"state": "suggest",    "value": {"mode": "same_referent_as_input", "aliases": ["Developer"]}}]}

A prediction is correct if it satisfies ANY allowed entry (lenient) or the
FIRST entry (strict). Value modes:

- exact / one_of: closed sets (used for acceptance -- value must be a scope item)
- empty: used for decline
- same_referent_as_input: open answer space, validated by a deterministic
  tier chain: (1) normalized equality (casefold, strip non-alphanumerics, so
  "Smart Watch" == "Smartwatch"), (2) alias table, (3) Levenshtein ratio >=
  fuzzy_threshold (default 0.85), (4) optional LLM-as-judge fallback
  (--judge, 3 votes majority, different model than the one under test).
  Preconditions: value must be clean (no noise chars) and not already in scope.

Legacy expected.csv files are still read as literal specs.

Run the scoring harness on a single case, a group, or everything:

    python evaluate.py not_mixed_tests/accept_with_renaming/2D
    python evaluate.py not_mixed_tests --repeats 5
    python evaluate.py . --repeats 3 --out results.json
    python evaluate.py not_mixed_tests/sugestion --judge --judge-model openai/gpt-4o-mini

Reports strict accuracy (primary spec), lenient accuracy (any allowed spec),
a confusion matrix over states, per-class precision/recall/F1, and which
validation tier resolved each correct verdict. Use --repeats to measure
output stability across runs.

## Results database
Add --db to any run to record every prediction in SQLite for querying and
visualization (confusion-matrix heatmaps, per-scenario accuracy, stability):

    python evaluate.py not_mixed_tests --repeats 5 --db results.sqlite

Tables: runs (one row per invocation: model, judge, repeats, timestamp) and
results (one row per prediction: scenario, domain, raw_value, gold_state,
pred_state, pred_value, strict, lenient, matched_tier). See queries.md for
ready-made SQL and seaborn heatmap snippets.

## Large-scale test data (large_tests/)
A generated dataset with 1000+ inputs per scenario (6188 total) across 10 entity
domains (countries, cities, animals, programming languages, colors, chemical
elements, fruits/vegetables, departments, currencies, job titles), plus a large
mixed scenario combining all four outcomes against one scope:

    accept_without_renaming  1336  (exact scope matches)
    accept_with_renaming     1336  (typos, casing, noise chars, aliases/abbreviations)
    suggestion               1117  (held-out same-category entities)
    decline                  1237  (gibberish, placeholders, free text, wrong category)
    mixed                    1162  (all four outcomes, combined scope)

Gold labels are expected.jsonl predicate specs. The dataset is fully
reproducible from a fixed seed: python generate_data.py (the generator lives
on the new_testing_data_generator branch).

NOTE: running evaluate.py on large_tests/ makes one API call per input value
(6188 calls with the default 2s delay is ~3.5h) -- evaluate a subfolder or a
single scenario when iterating.

Methodology note: validate the validator -- hand-label a stratified sample
(oversampling suggest cases), compare against the automated verdicts, and
report the agreement rate. If the judge tier is used, report judge-human
agreement separately and pin the judge model + prompt version.