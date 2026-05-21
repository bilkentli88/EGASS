from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# Dataset-level paired statistical comparison
# =============================================================================

DEFAULT_PROPOSED_MODEL = "Proposed_Unnormalized_Base"

DEFAULT_COMPARISON_MODELS: List[str] = [
    "PlainSSM",
    "NormGated",
    "Proposed_Unnormalized_Sparse_1e-5",
]

METRICS: List[str] = ["accuracy", "macro_f1"]

SUMMARY_COLUMNS = [
    "comparison",
    "metric",
    "n_datasets",
    "mean_difference",
    "median_difference",
    "wins",
    "ties",
    "losses",
    "wilcoxon_p_value",
]

DATASET_DETAIL_COLUMNS = [
    "dataset",
    "comparison_model",
    "proposed_model",
    "metric",
    "comparison_mean",
    "proposed_mean",
    "difference",
]


@dataclass
class StatConfig:
    input_detail_csv: Path
    results_dir: Path = Path("Results")
    proposed_model: str = DEFAULT_PROPOSED_MODEL
    comparison_models: List[str] = None
    output_prefix: str = "dataset_level_paired_statistical_tests"

    def __post_init__(self) -> None:
        if self.comparison_models is None:
            self.comparison_models = list(DEFAULT_COMPARISON_MODELS)


def safe_float(value) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        if value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def paired_wilcoxon_from_differences(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test on paired differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return np.nan

    if np.allclose(diff, 0.0):
        return 1.0

    if not SCIPY_AVAILABLE:
        return np.nan

    try:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def count_wins_ties_losses(diff: np.ndarray, atol: float = 1e-12) -> Tuple[int, int, int]:
    """Count wins, ties, and losses for Proposed minus comparison differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    wins = int(np.sum(diff > atol))
    ties = int(np.sum(np.abs(diff) <= atol))
    losses = int(np.sum(diff < -atol))

    return wins, ties, losses


def format_p_value(p_value: float) -> str:
    """Format p-values for LaTeX output."""
    if not np.isfinite(p_value):
        return "--"
    if p_value < 1e-3:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def latex_metric_name(metric: str) -> str:
    if metric == "accuracy":
        return "Accuracy"
    if metric == "macro_f1":
        return "Macro-F1"
    return metric


def latex_model_name(model_name: str) -> str:
    mapping = {
        "PlainSSM": "PlainSSM",
        "NormGated": "NormGated",
        "Proposed_Unnormalized_Sparse_1e-5": r"Sparse \(10^{-5}\)",
        "Proposed_Unnormalized_Base": "Proposed Base",
    }
    return mapping.get(model_name, model_name.replace("_", r"\_"))


def load_successful_detail_rows(input_detail_csv: Path) -> pd.DataFrame:
    """Load successful rows from the main aggregation-ablation detail CSV."""
    if not input_detail_csv.exists():
        raise FileNotFoundError(f"Input detail CSV not found: {input_detail_csv}")

    df = pd.read_csv(input_detail_csv)

    required = {"dataset", "model", "seed", "status", "accuracy", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = df[df["status"] == "success"].copy()

    for col in ["seed", "accuracy", "macro_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_dataset_level_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average each model's performance over seeds for each dataset."""
    group_cols = ["dataset", "model"]

    means = (
        df.groupby(group_cols, as_index=False)[METRICS]
        .mean()
        .sort_values(group_cols)
    )

    return means


def compute_statistical_tables(cfg: StatConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute dataset-level differences and summary paired tests."""
    df = load_successful_detail_rows(cfg.input_detail_csv)
    means = compute_dataset_level_means(df)

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for comparison_model in cfg.comparison_models:
        for metric in METRICS:
            pivot = means.pivot_table(
                index="dataset",
                columns="model",
                values=metric,
                aggfunc="first",
            )

            if cfg.proposed_model not in pivot.columns:
                raise ValueError(f"Proposed model not found in input CSV: {cfg.proposed_model}")

            if comparison_model not in pivot.columns:
                raise ValueError(f"Comparison model not found in input CSV: {comparison_model}")

            pair = pivot[[comparison_model, cfg.proposed_model]].dropna()
            comparison_values = pair[comparison_model].to_numpy(dtype=float)
            proposed_values = pair[cfg.proposed_model].to_numpy(dtype=float)
            differences = proposed_values - comparison_values

            wins, ties, losses = count_wins_ties_losses(differences)
            p_value = paired_wilcoxon_from_differences(differences)

            summary_rows.append({
                "comparison": f"{cfg.proposed_model} vs {comparison_model}",
                "metric": metric,
                "n_datasets": int(len(pair)),
                "mean_difference": float(np.mean(differences)) if len(differences) else np.nan,
                "median_difference": float(np.median(differences)) if len(differences) else np.nan,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_p_value": p_value,
            })

            for dataset, row in pair.iterrows():
                detail_rows.append({
                    "dataset": dataset,
                    "comparison_model": comparison_model,
                    "proposed_model": cfg.proposed_model,
                    "metric": metric,
                    "comparison_mean": float(row[comparison_model]),
                    "proposed_mean": float(row[cfg.proposed_model]),
                    "difference": float(row[cfg.proposed_model] - row[comparison_model]),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def write_latex_table(summary: pd.DataFrame, output_tex: Path) -> None:
    """Write the manuscript-style LaTeX table for Section 5.4."""
    comparison_order = {
        "PlainSSM": 0,
        "NormGated": 1,
        "Proposed_Unnormalized_Sparse_1e-5": 2,
    }

    def comparison_key(label: str) -> int:
        for model_name, order in comparison_order.items():
            if label.endswith(model_name):
                return order
        return 99

    summary = summary.copy()
    summary["comparison_order"] = summary["comparison"].map(comparison_key)
    summary["metric_order"] = summary["metric"].map({"accuracy": 0, "macro_f1": 1})
    summary = summary.sort_values(["comparison_order", "metric_order"])

    rows_by_comparison = {}
    for _, row in summary.iterrows():
        rows_by_comparison.setdefault(row["comparison"], {})[row["metric"]] = row

    with open(output_tex, "w", encoding="utf-8") as handle:
        handle.write(r"\begin{table}[t]" + "\n")
        handle.write(r"\centering" + "\n")
        handle.write(
            r"\caption{Dataset-level paired comparison between Proposed Base and matched variants over the selected UCR datasets. "
            r"Differences are computed using dataset-level means over five random seeds. "
            r"W/T/L denotes the number of datasets on which Proposed Base wins, ties, or loses against the comparison model.}" + "\n"
        )
        handle.write(r"\label{tab:paired_tests}" + "\n")
        handle.write(r"\resizebox{\textwidth}{!}{" + "\n")
        handle.write(r"\begin{tabular}{lcccccc}" + "\n")
        handle.write(r"\toprule" + "\n")
        handle.write(r"\multirow{2}{*}{Comparison} & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{Macro-F1} \\" + "\n")
        handle.write(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n")
        handle.write(r"& Mean diff. & W/T/L & \(p\)-value & Mean diff. & W/T/L & \(p\)-value \\" + "\n")
        handle.write(r"\midrule" + "\n")

        for comparison_label, metric_rows in rows_by_comparison.items():
            comparison_model = comparison_label.split(" vs ", 1)[1]
            acc = metric_rows.get("accuracy")
            f1 = metric_rows.get("macro_f1")

            if acc is None or f1 is None:
                continue

            handle.write(
                f"Proposed Base vs {latex_model_name(comparison_model)} "
                f"& {acc['mean_difference']:.3f} "
                f"& {int(acc['wins'])}/{int(acc['ties'])}/{int(acc['losses'])} "
                f"& {format_p_value(acc['wilcoxon_p_value'])} "
                f"& {f1['mean_difference']:.3f} "
                f"& {int(f1['wins'])}/{int(f1['ties'])}/{int(f1['losses'])} "
                f"& {format_p_value(f1['wilcoxon_p_value'])} \\\\\n"
            )

        handle.write(r"\bottomrule" + "\n")
        handle.write(r"\end{tabular}" + "\n")
        handle.write(r"}" + "\n")
        handle.write(r"\end{table}" + "\n")


def write_outputs(cfg: StatConfig, summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    """Write CSV and LaTeX outputs."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = cfg.results_dir / f"{cfg.output_prefix}_summary.csv"
    detail_csv = cfg.results_dir / f"{cfg.output_prefix}_dataset_details.csv"
    latex_tex = cfg.results_dir / f"{cfg.output_prefix}_table.tex"

    summary.to_csv(summary_csv, index=False, columns=SUMMARY_COLUMNS)
    detail.to_csv(detail_csv, index=False, columns=DATASET_DETAIL_COLUMNS)
    write_latex_table(summary, latex_tex)

    print("Dataset-level paired statistical testing completed.")
    print("Input detail CSV:", cfg.input_detail_csv)
    print("Summary CSV:", summary_csv)
    print("Dataset-detail CSV:", detail_csv)
    print("LaTeX table:", latex_tex)

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy is not available; Wilcoxon p-values were written as NaN.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level paired statistical comparisons from the "
            "multi-seed aggregation-ablation detail CSV."
        )
    )

    parser.add_argument(
        "--input-detail-csv",
        type=str,
        required=True,
        help="Path to multiseed_aggregation_ablation_detail_<timestamp>.csv.",
    )
    parser.add_argument("--results-dir", type=str, default="Results")
    parser.add_argument("--output-prefix", type=str, default="dataset_level_paired_statistical_tests")
    parser.add_argument("--proposed-model", type=str, default=DEFAULT_PROPOSED_MODEL)
    parser.add_argument(
        "--comparison-models",
        nargs="+",
        default=DEFAULT_COMPARISON_MODELS,
        help="Model names to compare against the proposed model.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = StatConfig(
        input_detail_csv=Path(args.input_detail_csv),
        results_dir=Path(args.results_dir),
        output_prefix=args.output_prefix,
        proposed_model=args.proposed_model,
        comparison_models=list(args.comparison_models),
    )

    summary, detail = compute_statistical_tables(cfg)
    write_outputs(cfg, summary, detail)


if __name__ == "__main__":
    main()
