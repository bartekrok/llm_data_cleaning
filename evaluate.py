"""Scoring harness for the LLM data-cleaning experiments.

Runs each test case N times (LLM output is stochastic even at low temperature),
compares predictions against gold labels in expected.csv, and reports:

  - strict accuracy  (prediction matches the primary gold label)
  - lenient accuracy (prediction matches the primary OR the acceptable
                      alternative label -- used for genuinely ambiguous items)
  - a 4-way confusion matrix over states (acceptance / suggest / decline / error)
  - per-class precision, recall and F1

expected.csv format (one per test folder, next to scope.csv / input_data.csv):

  raw_value,state,value,alt_state,alt_value

alt_state/alt_value are optional and encode an acceptable-answer set for
items where two labels are defensible (e.g. rename vs. suggest).

Usage:
  python evaluate.py not_mixed_tests/accept_with_renaming/2D
  python evaluate.py not_mixed_tests --repeats 5
  python evaluate.py . --repeats 3 --out results.json
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict

from script import clean_value_with_llm, load_scope

STATES = ["acceptance", "suggest", "decline", "error"]
SLEEP_BETWEEN_CALLS = 2  # seconds, mirrors script.py


def norm(s):
    return (s or "").strip().lower()


def load_expected(filepath):
    """Returns {raw_value(normalized): [(state, value), ...acceptable answers]}."""
    expected = {}
    with open(filepath, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answers = [(norm(row["state"]), norm(row["value"]))]
            if norm(row.get("alt_state")):
                answers.append((norm(row["alt_state"]), norm(row.get("alt_value"))))
            expected[norm(row["raw_value"])] = answers
    return expected


def matches(prediction, gold_answer):
    """A prediction matches a gold answer if the state agrees and, for
    non-decline states, the standardized value agrees (case-insensitive)."""
    pred_state, pred_value = prediction
    gold_state, gold_value = gold_answer
    if pred_state != gold_state:
        return False
    if gold_state == "decline":
        return True
    return pred_value == gold_value


def find_test_folders(root):
    """A test folder is any folder containing scope.csv + input_data.csv + expected.csv."""
    required = {"scope.csv", "input_data.csv", "expected.csv"}
    folders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if required.issubset(set(filenames)):
            folders.append(dirpath)
    return sorted(folders)


def evaluate_folder(folder, repeats):
    scope = load_scope(os.path.join(folder, "scope.csv"))
    expected = load_expected(os.path.join(folder, "expected.csv"))
    records = []

    with open(os.path.join(folder, "input_data.csv"), mode="r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if norm(r.get("raw_value"))]

    for row in rows:
        raw_value = row["raw_value"].strip()
        gold_answers = expected.get(norm(raw_value))
        if gold_answers is None:
            print(f"  WARNING: no gold label for '{raw_value}' in {folder}, skipping")
            continue

        for run in range(repeats):
            result = clean_value_with_llm(raw_value, scope)
            pred = (norm(result.get("state")), norm(result.get("value")))
            strict = matches(pred, gold_answers[0])
            lenient = any(matches(pred, g) for g in gold_answers)
            records.append({
                "folder": folder,
                "raw_value": raw_value,
                "run": run + 1,
                "gold_state": gold_answers[0][0],
                "gold_value": gold_answers[0][1],
                "ambiguous": len(gold_answers) > 1,
                "pred_state": pred[0] if pred[0] in STATES else "error",
                "pred_value": result.get("value", ""),
                "strict": strict,
                "lenient": lenient,
                "message": result.get("message", ""),
            })
            status = "OK " if lenient else "MISS"
            print(f"  [{status}] '{raw_value}' run {run + 1}: "
                  f"pred={pred[0]}/{result.get('value', '')} "
                  f"gold={gold_answers[0][0]}/{gold_answers[0][1]}")
            time.sleep(SLEEP_BETWEEN_CALLS)

    return records


def summarize(records):
    n = len(records)
    if n == 0:
        return {"n": 0}

    strict_acc = sum(r["strict"] for r in records) / n
    lenient_acc = sum(r["lenient"] for r in records) / n

    confusion = defaultdict(Counter)  # gold_state -> pred_state -> count
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
    return {
        "n": n,
        "strict_accuracy": round(strict_acc, 3),
        "lenient_accuracy": round(lenient_acc, 3),
        "ambiguous_items_n": len(ambiguous),
        "ambiguous_lenient_accuracy": (
            round(sum(r["lenient"] for r in ambiguous) / len(ambiguous), 3)
            if ambiguous else None
        ),
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
    parser = argparse.ArgumentParser(description="Score LLM data-cleaning predictions against expected.csv gold labels.")
    parser.add_argument("path", help="A test folder, or a root folder to scan recursively for test cases")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per value (default 1); use >1 to measure stability")
    parser.add_argument("--out", default=None, help="Optional path to write full results + summary as JSON")
    args = parser.parse_args()

    folders = find_test_folders(args.path)
    if not folders:
        print(f"Error: no test folders (scope.csv + input_data.csv + expected.csv) found under '{args.path}'")
        sys.exit(1)

    all_records = []
    for folder in folders:
        print(f"\n### {folder}")
        all_records.extend(evaluate_folder(folder, args.repeats))

    summary = summarize(all_records)
    print_summary(summary)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": all_records}, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
