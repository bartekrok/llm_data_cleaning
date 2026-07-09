# Results database: schema and example queries

Run the harness with `--db` to record every prediction:

    python evaluate.py not_mixed_tests --repeats 5 --db results.sqlite

## Schema

`runs` — one row per invocation: `run_id, timestamp, model, judge_enabled,
judge_model, repeats, path`. Compare models or prompt versions across runs.

`results` — one row per prediction: `run_id, scenario, domain, folder,
raw_value, run_n, gold_state, ambiguous, pred_state, pred_value,
strict, lenient, matched_tier, message`.

`strict`/`lenient` are 0/1 verdicts. `matched_tier` says which validation
tier confirmed a correct answer (`exact`, `normalized`, `alias`, `fuzzy`,
`judge`) or `none` for misses.

## Example queries

Confusion matrix (heatmap: pivot gold_state x pred_state):

    SELECT gold_state, pred_state, COUNT(*) AS n
    FROM results WHERE run_id = 1
    GROUP BY gold_state, pred_state;

Accuracy per scenario and domain (heatmap: pivot scenario x domain):

    SELECT scenario, domain,
           AVG(lenient) AS lenient_acc, AVG(strict) AS strict_acc, COUNT(*) AS n
    FROM results WHERE run_id = 1
    GROUP BY scenario, domain;

Strict-vs-lenient gap on ambiguous items:

    SELECT scenario, AVG(strict) AS strict_acc, AVG(lenient) AS lenient_acc
    FROM results WHERE run_id = 1 AND ambiguous = 1
    GROUP BY scenario;

Validation-tier distribution (how much of the score is deterministic):

    SELECT matched_tier, COUNT(*) AS n
    FROM results WHERE run_id = 1 AND lenient = 1
    GROUP BY matched_tier ORDER BY n DESC;

Stability across repeats (inputs whose predicted state flips between runs):

    SELECT folder, raw_value, COUNT(DISTINCT pred_state) AS n_states
    FROM results WHERE run_id = 1
    GROUP BY folder, raw_value HAVING n_states > 1;

Hardest individual inputs:

    SELECT folder, raw_value, AVG(lenient) AS acc, COUNT(*) AS n
    FROM results WHERE run_id = 1
    GROUP BY folder, raw_value ORDER BY acc ASC, n DESC LIMIT 25;

Model comparison across runs:

    SELECT r.run_id, u.model, u.timestamp, AVG(r.lenient) AS lenient_acc
    FROM results r JOIN runs u USING (run_id)
    GROUP BY r.run_id;

## Heatmap in Python

    import pandas as pd, sqlite3, seaborn as sns
    conn = sqlite3.connect("results.sqlite")

    cm = pd.read_sql("""SELECT gold_state, pred_state, COUNT(*) AS n
                        FROM results WHERE run_id = 1
                        GROUP BY 1, 2""", conn)
    sns.heatmap(cm.pivot(index="gold_state", columns="pred_state", values="n")
                  .fillna(0), annot=True, fmt=".0f")

    acc = pd.read_sql("""SELECT scenario, domain, AVG(lenient) AS acc
                         FROM results WHERE run_id = 1
                         GROUP BY 1, 2""", conn)
    sns.heatmap(acc.pivot(index="scenario", columns="domain", values="acc"),
                annot=True, fmt=".2f", vmin=0, vmax=1)
