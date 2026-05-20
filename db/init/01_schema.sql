-- Schema for the LLM data-cleaning experiment harness.
-- See plan §5.2 and §9.

CREATE TABLE IF NOT EXISTS test_cases (
    id              SERIAL PRIMARY KEY,
    test_name       TEXT NOT NULL UNIQUE,
    scenario        TEXT NOT NULL,
    raw_value       TEXT NOT NULL,
    scope           TEXT[] NOT NULL,
    expected_state  TEXT NOT NULL,
    expected_value  TEXT,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS test_cases_scenario_idx ON test_cases(scenario);

CREATE TABLE IF NOT EXISTS runs (
    id                    BIGSERIAL PRIMARY KEY,
    experiment_id         TEXT NOT NULL,
    experiment_name       TEXT NOT NULL,
    ts                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_name            TEXT NOT NULL,
    prompt_variant        TEXT NOT NULL,
    mode                  TEXT NOT NULL,
    temperature           REAL NOT NULL,
    repetition_idx        INT NOT NULL,
    test_case_id          INT REFERENCES test_cases(id) ON DELETE SET NULL,
    test_name             TEXT NOT NULL,
    raw_value             TEXT,
    scope_size            INT,
    actual_state          TEXT,
    actual_value          TEXT,
    actual_message        TEXT,
    actual_confidence     REAL,
    expected_state        TEXT NOT NULL,
    expected_value        TEXT,
    is_correct            BOOLEAN,
    failure_mode          TEXT,
    latency_ms            INT,
    tokens_in             INT,
    tokens_out            INT,
    cost_usd              NUMERIC(10, 6),
    response_format_used  TEXT,
    provider              TEXT,
    seed                  INT,
    raw_response          JSONB
);

CREATE INDEX IF NOT EXISTS runs_experiment_idx        ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS runs_experiment_name_idx   ON runs(experiment_name);
CREATE INDEX IF NOT EXISTS runs_model_idx             ON runs(model_name);
CREATE INDEX IF NOT EXISTS runs_test_case_idx         ON runs(test_case_id);
CREATE INDEX IF NOT EXISTS runs_prompt_variant_idx    ON runs(prompt_variant);
CREATE INDEX IF NOT EXISTS runs_failure_mode_idx      ON runs(failure_mode);

CREATE OR REPLACE VIEW runs_summary AS
SELECT
    experiment_id,
    experiment_name,
    model_name,
    prompt_variant,
    mode,
    temperature,
    COUNT(*)                                                   AS n_runs,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END)                AS n_correct,
    ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END)::numeric, 4) AS accuracy,
    SUM(CASE WHEN failure_mode = 'parse_error' THEN 1 ELSE 0 END) AS n_parse_errors,
    SUM(CASE WHEN failure_mode = 'api_error'   THEN 1 ELSE 0 END) AS n_api_errors,
    ROUND(AVG(latency_ms)::numeric, 1)                         AS avg_latency_ms,
    ROUND(SUM(cost_usd)::numeric, 4)                           AS total_cost_usd
FROM runs
GROUP BY experiment_id, experiment_name, model_name, prompt_variant, mode, temperature;
