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

Methodology note: validate the validator -- hand-label a stratified sample
(oversampling suggest cases), compare against the automated verdicts, and
report the agreement rate. If the judge tier is used, report judge-human
agreement separately and pin the judge model + prompt version.