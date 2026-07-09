"""Scoring harness for the LLM data-cleaning experiments.

Runs each test case N times (LLM output is stochastic even at low temperature),
validates predictions against gold specs, and reports:

  - strict accuracy   (prediction satisfies the PRIMARY allowed answer)
  - lenient accuracy  (prediction satisfies ANY allowed answer)
  - a 4-way confusion matrix over states (acceptance / suggest / decline / error)
  - per-class precision, recall and F1
  - which validation tier resolved each verdict

Gold format: expected.jsonl (one JSON object per line)
------------------------------------------------------
Enumerating correct answers does not work for open answer spaces (a suggested
value has a whole spectrum of correct surface forms). The gold file therefore
specifies PREDICATES, not literals:

  {"raw_value": "Dev",
   "allowed": [
     {"state": "acceptance", "value": {"mode": "one_of", "items": ["Software Engineer"]}},
     {"state": "suggest",    "value": {"mode": "same_referent_as_input",
                                       "aliases": ["Developer"]}}
   ]}

A prediction is correct if it satisfies ANY entry in "allowed"; "strict"
scoring uses only the first (primary) entry. Value modes:

  exact                  value == item (case/whitespace-insensitive)
  one_of                 value is one of "items"
  empty                  value must be "" (used for decline)
  same_referent_as_input value must be a cleaned surface form of the input
                         entity. Validated by a deterministic tier chain:
                           1. normalized   casefold + strip non-alphanumerics
                           2. alias        matches an entry in "aliases"
                           3. fuzzy        Levenshtein ratio >= fuzzy_threshold
                                           (default 0.85)
                           4. judge        optional LLM-as-judge fallback,
                                           only with --judge (3 votes, majority)
                         Preconditions: value must be clean (no noise chars)
                         and, by default, NOT already in scope
                         ("not_in_scope": false to disable).

Legacy expected.csv files (raw_value,state,value,alt_state,alt_value) are
still read and interpreted as literal one_of/empty specs.

Usage:
  python evaluate.py not_mixed_tests/accept_with_renaming/2D
  python evaluate.py not_mixed_tests --repeats 5
  python evaluate.py . --repeats 3 --out results.json
  python evaluate.py sugestion_case --judge --judge-model openai/gpt-4o-mini
"""

import argparse
import csv
import datetime
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict

from script import clean_value_with_llm, load_scope, API_URL, API_KEY, MODEL

STATES = ["acceptance", "suggest", "decline", "error"]
SLEEP_BETWEEN_CALLS = 2  # seconds, mirrors script.py
DEFAULT_FUZZY_THRESHOLD = 0.85
JUDGE_VOTES = 3
CLEAN_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+#&'()/-]{0,79}$")


def norm(s):
    return (s or "").strip().lower()


def ref_norm(s):
    """Normalization for same-referent comparison: casefold and strip
    everything that is not a letter or digit ('Smart Watch' == 'smartwatch').
    '#' and '+' are kept -- they are distinctive, not noise ('C' vs 'C++'
    vs 'C#' must not collide)."""
    return re.sub(r"[^a-z0-9#+]", "", (s or "").casefold())


def lev_ratio(a, b):
    """Levenshtein similarity ratio in [0, 1]."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# ---------------------------------------------------------------------------
# LLM-as-judge (optional tier 4)
# ---------------------------------------------------------------------------

class Judge:
    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled
        self.cache = {}
        self.calls = 0

    def same_referent(self, raw_value, value):
        """Majority vote over JUDGE_VOTES calls: do raw_value and value name
        the same real-world entity (differing only in surface form)?"""
        if not self.enabled:
            return False
        key = (ref_norm(raw_value), ref_norm(value))
        if key in self.cache:
            return self.cache[key]

        import requests  # already a project dependency via script.py
        system = (
            "You are a strict evaluation judge. Answer ONLY with JSON: "
            '{"verdict": "yes"} or {"verdict": "no"}.\n'
            "Question: do the two strings name the SAME real-world entity, "
            "differing only in surface form (typo, casing, abbreviation, noise "
            "characters, well-established alias)? Related-but-distinct entities "
            'of the same category are "no".'
        )
        votes = 0
        for _ in range(JUDGE_VOTES):
            try:
                r = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": [
                              {"role": "system", "content": system},
                              {"role": "user",
                               "content": f'String A: "{raw_value}"\nString B: "{value}"'}],
                          "temperature": 0.0},
                    timeout=60)
                r.raise_for_status()
                self.calls += 1
                out = r.json()["choices"][0]["message"]["content"].strip()
                out = out.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                if json.loads(out).get("verdict") == "yes":
                    votes += 1
            except Exception:
                pass
            time.sleep(SLEEP_BETWEEN_CALLS)
        verdict = votes > JUDGE_VOTES // 2
        self.cache[key] = verdict
        return verdict


# ---------------------------------------------------------------------------
# Gold specs
# ---------------------------------------------------------------------------

def load_expected(folder):
    """Returns {norm(raw_value): [allowed_entry, ...]} from expected.jsonl,
    falling back to legacy expected.csv."""
    jsonl_path = os.path.join(folder, "expected.jsonl")
    if os.path.exists(jsonl_path):
        expected = {}
        with open(jsonl_path, mode="r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                expected[norm(obj["raw_value"])] = obj["allowed"]
        return expected

    expected = {}
    with open(os.path.join(folder, "expected.csv"), mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            allowed = [_csv_entry(row["state"], row["value"])]
            if norm(row.get("alt_state")):
                allowed.append(_csv_entry(row["alt_state"], row.get("alt_value")))
            expected[norm(row["raw_value"])] = allowed
    return expected


def _csv_entry(state, value):
    state = norm(state)
    if state == "decline":
        return {"state": "decline", "value": {"mode": "empty"}}
    return {"state": state, "value": {"mode": "one_of", "items": [value or ""]}}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_value(mode_obj, pred_value, raw_value, scope, judge):
    """Returns (ok: bool, tier: str)."""
    mode = mode_obj.get("mode", "exact")

    if mode == "empty":
        return (norm(pred_value) == "", "exact")

    if mode == "exact":
        return (norm(pred_value) == norm(mode_obj.get("item", "")), "exact")

    if mode == "one_of":
        ok = norm(pred_value) in {norm(i) for i in mode_obj.get("items", [])}
        return (ok, "exact")

    if mode == "same_referent_as_input":
        value = (pred_value or "").strip()
        # precondition: a cleaned, plausible value
        if not CLEAN_VALUE_RE.match(value):
            return (False, "cleanliness")
        # precondition: must not already be in scope (it would be acceptance)
        if mode_obj.get("not_in_scope", True):
            if ref_norm(value) in {ref_norm(s) for s in scope}:
                return (False, "not_in_scope")
        # tier 1: normalized equality with the input
        if ref_norm(value) == ref_norm(raw_value):
            return (True, "normalized")
        # tier 2: alias table
        aliases = {ref_norm(a) for a in mode_obj.get("aliases", [])}
        if ref_norm(value) in aliases:
            return (True, "alias")
        # tier 3: fuzzy match against the input
        threshold = mode_obj.get("fuzzy_threshold", DEFAULT_FUZZY_THRESHOLD)
        if lev_ratio(ref_norm(value), ref_norm(raw_value)) >= threshold:
            return (True, "fuzzy")
        # tier 4: LLM-as-judge (only when --judge is set)
        if judge.enabled and judge.same_referent(raw_value, value):
            return (True, "judge")
        return (False, "judge" if judge.enabled else "fuzzy")

    raise ValueError(f"Unknown value mode: {mode}")


def validate_entry(entry, pred_state, pred_value, raw_value, scope, judge):
    if pred_state != norm(entry["state"]):
        return (False, "state")
    return validate_value(entry["value"], pred_value, raw_value, scope, judge)


# ---------------------------------------------------------------------------
# Results database (--db): one row per prediction, queryable for heatmaps etc.
# ---------------------------------------------------------------------------

KNOWN_ROOTS = {"large_tests", "not_mixed_tests", "mixed_tests"}


def split_scenario(folder):
    """'large_tests/suggestion/animals' -> ('suggestion', 'animals')."""
    parts = [p for p in os.path.normpath(folder).split(os.sep) if p not in (".", "")]
    for i, p in enumerate(parts):
        if p in KNOWN_ROOTS:
            scenario = parts[i + 1] if len(parts) > i + 1 else ""
            domain = parts[i + 2] if len(parts) > i + 2 else ""
            return scenario, domain
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return (parts[0] if parts else "", "")


def init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            model         TEXT NOT NULL,
            judge_enabled INTEGER NOT NULL,
            judge_model   TEXT,
            repeats       INTEGER NOT NULL,
            path          TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       INTEGER NOT NULL REFERENCES runs(run_id),
            scenario     TEXT,
            domain       TEXT,
            folder       TEXT NOT NULL,
            raw_value    TEXT NOT NULL,
            run_n        INTEGER NOT NULL,
            gold_state   TEXT NOT NULL,
            ambiguous    INTEGER NOT NULL,
            pred_state   TEXT NOT NULL,
            pred_value   TEXT,
            strict       INTEGER NOT NULL,
            lenient      INTEGER NOT NULL,
            matched_tier TEXT,
            message      TEXT
        )""")
    conn.commit()
    return conn


def start_run(conn, args):
    cur = conn.execute(
        "INSERT INTO runs (timestamp, model, judge_enabled, judge_model, repeats, path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(timespec="seconds"), MODEL,
         int(args.judge), args.judge_model if args.judge else None,
         args.repeats, args.path))
    conn.commit()
    return cur.lastrowid


def save_records(conn, run_id, records):
    rows = []
    for r in records:
        scenario, domain = split_scenario(r["folder"])
        rows.append((run_id, scenario, domain, r["folder"], r["raw_value"],
                     r["run"], r["gold_state"], int(r["ambiguous"]),
                     r["pred_state"], r["pred_value"], int(r["strict"]),
                     int(r["lenient"]), r["matched_tier"], r["message"]))
    conn.executemany(
        "INSERT INTO results (run_id, scenario, domain, folder, raw_value, run_n, "
        "gold_state, ambiguous, pred_state, pred_value, strict, lenient, "
        "matched_tier, message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def find_test_folders(root):
    folders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        names = set(filenames)
        if {"scope.csv", "input_data.csv"}.issubset(names) and \
                ("expected.jsonl" in names or "expected.csv" in names):
            folders.append(dirpath)
    return sorted(folders)


def evaluate_folder(folder, repeats, judge):
    scope = load_scope(os.path.join(folder, "scope.csv"))
    expected = load_expected(folder)
    records = []

    with open(os.path.join(folder, "input_data.csv"), mode="r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if norm(r.get("raw_value"))]

    for row in rows:
        raw_value = row["raw_value"].strip()
        allowed = expected.get(norm(raw_value))
        if allowed is None:
            print(f"  WARNING: no gold spec for '{raw_value}' in {folder}, skipping")
            continue

        for run in range(repeats):
            result = clean_value_with_llm(raw_value, scope)
            pred_state = norm(result.get("state"))
            if pred_state not in STATES:
                pred_state = "error"
            pred_value = result.get("value", "")

            strict, _ = validate_entry(allowed[0], pred_state, pred_value,
                                       raw_value, scope, judge)
            lenient, tier = False, "none"
            for entry in allowed:
                ok, t = validate_entry(entry, pred_state, pred_value,
                                       raw_value, scope, judge)
                if ok:
                    lenient, tier = True, t
                    break

            records.append({
                "folder": folder,
                "raw_value": raw_value,
                "run": run + 1,
                "gold_state": norm(allowed[0]["state"]),
                "ambiguous": len(allowed) > 1,
                "pred_state": pred_state,
                "pred_value": pred_value,
                "strict": strict,
                "lenient": lenient,
                "matched_tier": tier if lenient else "none",
                "message": result.get("message", ""),
            })
            status = "OK " if lenient else "MISS"
            print(f"  [{status}] '{raw_value}' run {run + 1}: "
                  f"pred={pred_state}/{pred_value} "
                  f"gold={allowed[0]['state']} tier={tier if lenient else '-'}")
            time.sleep(SLEEP_BETWEEN_CALLS)

    return records


def summarize(records, judge=None):
    n = len(records)
    if n == 0:
        return {"n": 0}

    strict_acc = sum(r["strict"] for r in records) / n
    lenient_acc = sum(r["lenient"] for r in records) / n

    confusion = defaultdict(Counter)
    for r in records:
        confusion[r["gold_state"]][r["pred_state"]] += 1

    per_class = {}
    for cls in STATES:
        tp = confusion[cls][cls]
        fn = sum(confusion[cls][p] for p in STATES if p != cls)
        fp = sum(confusion[g][cls] for g in STATES if g != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = tp + fn
        if support or fp:
            per_class[cls] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "support": support,
            }

    ambiguous = [r for r in records if r["ambiguous"]]
    tiers = Counter(r["matched_tier"] for r in records if r["lenient"])
    return {
        "n": n,
        "strict_accuracy": round(strict_acc, 3),
        "lenient_accuracy": round(lenient_acc, 3),
        "ambiguous_items_n": len(ambiguous),
        "ambiguous_lenient_accuracy": (
            round(sum(r["lenient"] for r in ambiguous) / len(ambiguous), 3)
            if ambiguous else None
        ),
        "validation_tiers": dict(tiers),
        "judge_calls": judge.calls if judge else 0,
        "confusion_matrix": {g: dict(c) for g, c in confusion.items()},
        "per_class": per_class,
    }


def print_summary(summary):
    print("\n" + "=" * 60)
    print(f"Evaluations:        {summary['n']}")
    print(f"Strict accuracy:    {summary['strict_accuracy']:.1%}")
    print(f"Lenient accuracy:   {summary['lenient_accuracy']:.1%}")
    if summary["ambiguous_items_n"]:
        print(f"Ambiguous items:    {summary['ambiguous_items_n']} "
              f"(lenient acc: {summary['ambiguous_lenient_accuracy']:.1%})")
    if summary["validation_tiers"]:
        tiers = ", ".join(f"{k}={v}" for k, v in sorted(summary["validation_tiers"].items()))
        print(f"Verdict tiers:      {tiers}")
    if summary["judge_calls"]:
        print(f"Judge calls:        {summary['judge_calls']}")

    print("\nConfusion matrix (rows = gold, cols = predicted):")
    header = "".join(f"{s:>12}" for s in STATES)
    print(f"{'':>12}{header}")
    for gold in STATES:
        row = summary["confusion_matrix"].get(gold)
        if not row:
            continue
        cells = "".join(f"{row.get(p, 0):>12}" for p in STATES)
        print(f"{gold:>12}{cells}")

    print("\nPer-class metrics:")
    for cls, m in summary["per_class"].items():
        print(f"  {cls:<12} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  support={m['support']}")


def main():
    parser = argparse.ArgumentParser(description="Score LLM data-cleaning predictions against gold specs.")
    parser.add_argument("path", help="A test folder, or a root folder to scan recursively for test cases")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per value (default 1); use >1 to measure stability")
    parser.add_argument("--judge", action="store_true",
                        help="Enable the LLM-as-judge fallback tier for same_referent_as_input specs")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini",
                        help="Judge model (should differ from the model under test)")
    parser.add_argument("--out", default=None, help="Optional path to write full results + summary as JSON")
    parser.add_argument("--db", default=None,
                        help="Optional SQLite file; appends one 'runs' row and one 'results' row "
                             "per prediction for post-run analysis (see queries.md)")
    args = parser.parse_args()

    folders = find_test_folders(args.path)
    if not folders:
        print(f"Error: no test folders (scope.csv + input_data.csv + expected.jsonl/csv) found under '{args.path}'")
        sys.exit(1)

    judge = Judge(args.judge_model, enabled=args.judge)
    conn = run_id = None
    if args.db:
        conn = init_db(args.db)
        run_id = start_run(conn, args)
        print(f"Recording to {args.db} as run_id={run_id}")

    all_records = []
    for folder in folders:
        print(f"\n### {folder}")
        records = evaluate_folder(folder, args.repeats, judge)
        if conn:
            save_records(conn, run_id, records)  # incremental: partial runs are kept
        all_records.extend(records)

    summary = summarize(all_records, judge)
    print_summary(summary)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": all_records}, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
