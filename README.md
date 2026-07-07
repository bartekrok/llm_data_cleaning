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
Each test folder contains an expected.csv gold file:

    raw_value,state,value,alt_state,alt_value

alt_state/alt_value encode an acceptable-answer set for genuinely ambiguous items
(e.g. "Dev" -> acceptance/Software Engineer OR suggest/Developer).

Run the scoring harness on a single case, a group, or everything:

    python evaluate.py not_mixed_tests/accept_with_renaming/2D
    python evaluate.py not_mixed_tests --repeats 5
    python evaluate.py . --repeats 3 --out results.json

Reports strict accuracy (primary label), lenient accuracy (any acceptable label),
a confusion matrix over states, and per-class precision/recall/F1. Use --repeats
to measure output stability across runs.