"""Metric computation and figure generation for the paper.

Reads from the `runs` table (and the `test_cases` table for context),
writes one CSV per metric and one PNG per figure into
    reports/<experiment-id_or_name>/

Metrics implemented (mapped to the RQs in the plan):

  RQ1: per-model x per-scenario accuracy heatmap, confusion matrix
       per (model, scenario) pair
  RQ2: per-model x per-prompt accuracy matrix
  RQ3: single vs batch_* accuracy comparison; scope-size scaling curves
  RQ4: LLMs vs baselines bar chart with cost overlay
  RQ5: variance: std-dev of correctness across reps, JSON-compliance
       rate, failure-mode taxonomy, reliability diagram + ECE for
       calibration runs
  Inter-model agreement: Cohen's kappa pairwise + McNemar's test

The module is designed to be tolerant: when a particular metric has
insufficient data (e.g. no batch runs in this experiment) it logs and
moves on.
"""

from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from llm_cleaner.etl import connect


def _safe_savefig(fig, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            pass


def _load_runs(
    experiment_id: Optional[str],
    experiment_name: Optional[str],
) -> pd.DataFrame:
    where = []
    params: list[Any] = []
    if experiment_id:
        where.append("experiment_id = %s")
        params.append(experiment_id)
    if experiment_name:
        where.append("experiment_name = %s")
        params.append(experiment_name)
    sql = (
        "SELECT id, experiment_id, experiment_name, ts, model_name, prompt_variant, mode, "
        "temperature, repetition_idx, test_case_id, test_name, raw_value, scope_size, "
        "actual_state, actual_value, actual_message, actual_confidence, "
        "expected_state, expected_value, is_correct, failure_mode, "
        "latency_ms, tokens_in, tokens_out, cost_usd, response_format_used, provider, seed "
        "FROM runs"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _load_test_cases() -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id AS test_case_id, test_name, scenario, expected_state, expected_value "
            "FROM test_cases"
        )
        cols = [c.name for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _accuracy_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False).agg(
        n=("is_correct", "size"),
        correct=("is_correct", "sum"),
    ).reset_index()
    g["accuracy"] = g["correct"] / g["n"].clip(lower=1)
    return g


def _heatmap(df: pd.DataFrame, index: str, columns: str, values: str, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pivot.columns) + 4), max(3, 0.45 * len(pivot.index) + 2)))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not (val is None or (isinstance(val, float) and math.isnan(val))):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=values)
    _safe_savefig(fig, path)


def _confusion_matrix(df: pd.DataFrame, path: Path, by_model: bool = True) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import numpy as np

    states = ["acceptance", "decline", "suggest"]
    rows = []
    groups = df.groupby("model_name") if by_model else [("all", df)]

    for model, sub in groups:
        cm = pd.crosstab(sub["expected_state"], sub["actual_state"].fillna("none"))
        cm = cm.reindex(index=states, fill_value=0)
        for col in states + ["none"]:
            if col not in cm.columns:
                cm[col] = 0
        cm = cm[states + ["none"]]
        for exp_state, row in cm.iterrows():
            for act_state, count in row.items():
                rows.append(
                    {"model_name": model, "expected_state": exp_state, "actual_state": act_state, "count": count}
                )

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels(cm.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("predicted")
        ax.set_ylabel("expected")
        ax.set_title(f"Confusion: {model}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm.values[i, j]), ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        safe_model = model.replace("/", "_")
        _safe_savefig(fig, path.with_name(f"confusion_{safe_model}.png"))

    return pd.DataFrame(rows)


def _per_state_prf(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    states = ["acceptance", "decline", "suggest"]
    for model, sub in df.groupby("model_name"):
        for s in states:
            tp = int(((sub["actual_state"] == s) & (sub["expected_state"] == s)).sum())
            fp = int(((sub["actual_state"] == s) & (sub["expected_state"] != s)).sum())
            fn = int(((sub["actual_state"] != s) & (sub["expected_state"] == s)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            out_rows.append(
                {
                    "model_name": model,
                    "state": s,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )
    return pd.DataFrame(out_rows)


def _pairwise_kappa(df: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa on the actual_state column between every model pair."""
    from sklearn.metrics import cohen_kappa_score

    pivot = (
        df.groupby(["model_name", "test_name", "repetition_idx"])["actual_state"]
        .first()
        .unstack(level="model_name")
    )
    models = list(pivot.columns)
    rows = []
    for a, b in combinations(models, 2):
        joined = pivot[[a, b]].dropna()
        if joined.empty:
            kappa = float("nan")
        else:
            kappa = cohen_kappa_score(joined[a].astype(str), joined[b].astype(str))
        rows.append({"model_a": a, "model_b": b, "n": len(joined), "cohen_kappa": round(float(kappa), 4) if kappa == kappa else None})
    return pd.DataFrame(rows)


def _pairwise_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """McNemar's test on (is_correct) between every model pair, paired by (test_name, repetition_idx)."""
    from statsmodels.stats.contingency_tables import mcnemar

    pivot = (
        df.groupby(["model_name", "test_name", "repetition_idx"])["is_correct"]
        .first()
        .unstack(level="model_name")
    )
    models = list(pivot.columns)
    rows = []
    for a, b in combinations(models, 2):
        joined = pivot[[a, b]].dropna()
        if joined.empty:
            rows.append({"model_a": a, "model_b": b, "n": 0, "b": 0, "c": 0, "stat": None, "p_value": None})
            continue
        a_ok = joined[a].astype(bool)
        b_ok = joined[b].astype(bool)
        b_only_correct = int((a_ok & ~b_ok).sum())
        c_only_correct = int((~a_ok & b_ok).sum())
        table = [[int((a_ok & b_ok).sum()), b_only_correct], [c_only_correct, int((~a_ok & ~b_ok).sum())]]
        try:
            res = mcnemar(table, exact=True)
            stat = res.statistic
            p = res.pvalue
        except Exception:
            stat, p = None, None
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n": len(joined),
                "a_only_correct": b_only_correct,
                "b_only_correct": c_only_correct,
                "stat": round(float(stat), 4) if isinstance(stat, (int, float)) and stat == stat else None,
                "p_value": round(float(p), 6) if isinstance(p, (int, float)) and p == p else None,
            }
        )
    return pd.DataFrame(rows)


def _variance_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model_name", "test_name", "temperature"])
    rows = []
    for (model, case, temp), sub in g:
        if len(sub) <= 1:
            continue
        states = sub["actual_state"].fillna("none").tolist()
        most_common = max(set(states), key=states.count)
        agreement = sum(1 for s in states if s == most_common) / len(states)
        correctness_std = float(sub["is_correct"].astype(float).std(ddof=0))
        rows.append(
            {
                "model_name": model,
                "test_name": case,
                "temperature": temp,
                "n_reps": len(sub),
                "agreement_rate": round(agreement, 4),
                "correctness_std": round(correctness_std, 4),
                "mean_accuracy": round(float(sub["is_correct"].astype(float).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _failure_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df[~df["is_correct"]]
        .groupby(["model_name", "failure_mode"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["model_name", "count"], ascending=[True, False])
    )
    return out


def _cost_and_latency(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model_name").agg(
        n=("id", "count"),
        accuracy=("is_correct", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        avg_cost_per_call=("cost_usd", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda s: float(pd.Series(s).quantile(0.95)) if len(s) else None),
        avg_tokens_in=("tokens_in", "mean"),
        avg_tokens_out=("tokens_out", "mean"),
    ).reset_index()
    return g


def _cost_vs_accuracy_plot(df_summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if df_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df_summary.dropna(subset=["accuracy"]).copy()
    sub["total_cost_usd"] = sub["total_cost_usd"].fillna(0.0)
    ax.scatter(sub["total_cost_usd"], sub["accuracy"], s=80)
    for _, r in sub.iterrows():
        ax.annotate(r["model_name"], (r["total_cost_usd"], r["accuracy"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. cost Pareto")
    ax.grid(True, alpha=0.3)
    _safe_savefig(fig, path)


def _reliability_diagram(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    sub = df[df["actual_confidence"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    for model, msub in sub.groupby("model_name"):
        ece = 0.0
        n_total = len(msub)
        fig, ax = plt.subplots(figsize=(6, 5))
        xs, ys, sizes = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            bin_mask = (msub["actual_confidence"] >= lo) & (msub["actual_confidence"] < hi)
            n_bin = int(bin_mask.sum())
            if n_bin == 0:
                rows.append({"model_name": model, "bin_low": lo, "bin_high": hi, "n": 0, "avg_confidence": None, "accuracy": None})
                continue
            avg_conf = float(msub.loc[bin_mask, "actual_confidence"].mean())
            acc = float(msub.loc[bin_mask, "is_correct"].astype(float).mean())
            rows.append({"model_name": model, "bin_low": lo, "bin_high": hi, "n": n_bin, "avg_confidence": round(avg_conf, 4), "accuracy": round(acc, 4)})
            ece += (n_bin / n_total) * abs(avg_conf - acc)
            xs.append(avg_conf)
            ys.append(acc)
            sizes.append(n_bin)

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        if xs:
            ax.scatter(xs, ys, s=[max(20, n * 2) for n in sizes])
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability diagram: {model} (ECE={ece:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        safe_model = model.replace("/", "_")
        _safe_savefig(fig, path.with_name(f"reliability_{safe_model}.png"))

    return pd.DataFrame(rows)


def analyze_experiment(
    *,
    experiment_id: Optional[str],
    experiment_name: Optional[str],
    out_dir: Path,
) -> None:
    df = _load_runs(experiment_id, experiment_name)
    if df.empty:
        print("No runs found for the given filters.")
        return

    label = experiment_id or experiment_name or "all"
    out_dir = out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {len(df)} runs into {out_dir}")

    df = df.merge(_load_test_cases()[["test_case_id", "scenario"]], on="test_case_id", how="left")
    df["scenario"] = df["scenario"].fillna("unknown")

    df.to_csv(out_dir / "runs_raw.csv", index=False)

    acc_overall = _accuracy_table(df, ["model_name"])
    acc_overall.to_csv(out_dir / "accuracy_overall.csv", index=False)

    acc_model_scenario = _accuracy_table(df, ["model_name", "scenario"])
    acc_model_scenario.to_csv(out_dir / "accuracy_by_model_scenario.csv", index=False)
    _heatmap(
        acc_model_scenario,
        index="model_name",
        columns="scenario",
        values="accuracy",
        title="Accuracy: model x scenario",
        path=out_dir / "accuracy_heatmap_model_scenario.png",
    )

    if df["prompt_variant"].nunique() > 1:
        acc_model_prompt = _accuracy_table(df, ["model_name", "prompt_variant"])
        acc_model_prompt.to_csv(out_dir / "accuracy_by_model_prompt.csv", index=False)
        _heatmap(
            acc_model_prompt,
            index="model_name",
            columns="prompt_variant",
            values="accuracy",
            title="Accuracy: model x prompt variant",
            path=out_dir / "accuracy_heatmap_model_prompt.png",
        )

    if df["mode"].nunique() > 1:
        acc_mode = _accuracy_table(df, ["model_name", "mode"])
        acc_mode.to_csv(out_dir / "accuracy_by_mode.csv", index=False)
        _heatmap(
            acc_mode,
            index="model_name",
            columns="mode",
            values="accuracy",
            title="Accuracy: model x mode (single vs batch_*)",
            path=out_dir / "accuracy_heatmap_mode.png",
        )

    if df["scope_size"].notna().sum() and df["scope_size"].nunique() > 1:
        acc_scope = (
            df.groupby(["model_name", "scope_size"])["is_correct"]
            .mean()
            .reset_index()
            .rename(columns={"is_correct": "accuracy"})
        )
        acc_scope.to_csv(out_dir / "accuracy_by_scope_size.csv", index=False)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            for model, sub in acc_scope.groupby("model_name"):
                ax.plot(sub["scope_size"], sub["accuracy"], marker="o", label=model)
            ax.set_xscale("log")
            ax.set_xlabel("Scope size")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy vs scope size")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _safe_savefig(fig, out_dir / "accuracy_vs_scope_size.png")
        except ImportError:
            pass

    _confusion_matrix(df, out_dir / "confusion.png")

    prf = _per_state_prf(df)
    prf.to_csv(out_dir / "precision_recall_f1.csv", index=False)

    if df["model_name"].nunique() > 1:
        kappa = _pairwise_kappa(df)
        kappa.to_csv(out_dir / "cohen_kappa_pairs.csv", index=False)

        mcnemar = _pairwise_mcnemar(df)
        mcnemar.to_csv(out_dir / "mcnemar_pairs.csv", index=False)

    if (df.groupby(["model_name", "test_name", "temperature"]).size() > 1).any():
        variance = _variance_table(df)
        variance.to_csv(out_dir / "variance_per_cell.csv", index=False)

    failure = _failure_taxonomy(df)
    failure.to_csv(out_dir / "failure_modes.csv", index=False)

    cost_lat = _cost_and_latency(df)
    cost_lat.to_csv(out_dir / "cost_latency.csv", index=False)
    _cost_vs_accuracy_plot(cost_lat, out_dir / "cost_vs_accuracy.png")

    if df["actual_confidence"].notna().any():
        reliability = _reliability_diagram(df, out_dir / "reliability.png")
        if not reliability.empty:
            reliability.to_csv(out_dir / "reliability_bins.csv", index=False)

    json_compliance = (
        df.assign(parse_error=df["failure_mode"].fillna("").eq("parse_error"))
        .groupby("model_name")["parse_error"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "n_parse_errors", "count": "n_runs"})
    )
    json_compliance["json_ok_rate"] = 1 - json_compliance["n_parse_errors"] / json_compliance["n_runs"].clip(lower=1)
    json_compliance.to_csv(out_dir / "json_compliance.csv", index=False)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "n_runs": int(len(df)),
                "n_models": int(df["model_name"].nunique()),
                "n_test_cases": int(df["test_name"].nunique()),
                "overall_accuracy": float(df["is_correct"].astype(float).mean()),
                "total_cost_usd": float(df["cost_usd"].fillna(0).sum()),
            },
            f,
            indent=2,
        )

    print("Generated:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")
