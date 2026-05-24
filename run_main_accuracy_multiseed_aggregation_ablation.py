from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from process_datasets import load_dataset, get_integer_labels_from_onehot


# =============================================================================
# Experiment metadata
# =============================================================================

DEFAULT_DATASETS: List[str] = [
    "ECG5000",
    "Wafer",
    "ElectricDevices",
    "FaceAll",
    "PhalangesOutlinesCorrect",
    "CricketX",
    "SwedishLeaf",
    "UWaveGestureLibraryX",
    "Yoga",
    "Earthquakes",
]

DEFAULT_SEEDS: List[int] = [2025, 2026, 2027, 2028, 2029]


DETAIL_COLUMNS = [
    "dataset",
    "model",
    "model_type",
    "pooling_type",
    "seed",
    "status",
    "error_message",
    "accuracy",
    "macro_f1",
    "val_accuracy",
    "val_macro_f1",
    "best_epoch",
    "epochs_run",
    "train_time_sec",
    "inference_time_sec",
    "n_params",
    "batch_size",
    "lr",
    "max_epochs",
    "patience",
    "hidden_dim",
    "latent_dim",
    "clip_grad_max_norm",
    "ce_loss",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "model_type",
    "pooling_type",
    "n_success",
    "n_failed",
    "accuracy_mean",
    "accuracy_std",
    "accuracy_min",
    "accuracy_max",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_f1_min",
    "macro_f1_max",
    "train_time_mean_sec",
    "inference_time_mean_sec",
    "alpha_mean_mean",
    "alpha_std_mean",
]


MODEL_CONFIGS = [
    {
        "model_name": "PlainSSM",
        "model_type": "plain",
        "pooling_type": "mean",
    },
    {
        "model_name": "NormGated",
        "model_type": "gated",
        "pooling_type": "normalized",
    },
    {
        "model_name": "Proposed_Unnormalized_Base",
        "model_type": "gated",
        "pooling_type": "unnormalized",
    },
]


@dataclass
class TrainConfig:
    """Configuration for the multi-seed UCR aggregation ablation experiment."""

    dataset_names: List[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))

    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 50
    patience: int = 10
    hidden_dim: int = 64
    latent_dim: int = 64

    clip_grad_max_norm: float = 0.5
    val_size: float = 0.2
    use_validation_split: bool = True

    results_dir: Path = Path("Results")
    detail_csv: str = ""
    summary_csv: str = ""
    device: torch.device = torch.device("cpu")


# =============================================================================
# Utilities
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    """Resolve requested computation device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return requested


def set_global_seed(seed: int) -> None:
    """Set NumPy/PyTorch random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_results_csv_names(results_dir: Path) -> Tuple[str, str]:
    """Create timestamped output file names."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_csv = results_dir / f"multiseed_aggregation_ablation_detail_{timestamp}.csv"
    summary_csv = results_dir / f"multiseed_aggregation_ablation_summary_{timestamp}.csv"

    return str(detail_csv), str(summary_csv)


def append_row_to_csv(csv_path: str | Path, row: Dict, columns: List[str]) -> None:
    """Append one row to a CSV file, creating the header if needed."""
    csv_path = Path(csv_path)
    exists = csv_path.exists()
    full_row = {col: row.get(col, "") for col in columns}

    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_std(values: np.ndarray) -> float:
    """Sample standard deviation with safe behavior for short vectors."""
    values = values[~np.isnan(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def safe_mean(values: np.ndarray) -> float:
    """Mean with safe behavior for empty vectors."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def to_float(value) -> float:
    """Convert a CSV value to float, returning NaN on failure."""
    try:
        if value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def create_train_val_test_loaders(
    X_train_full: torch.Tensor,
    y_train_full: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    seed: int,
    val_size: float,
    use_validation_split: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    X_train_full = X_train_full.detach().cpu()
    y_train_full = y_train_full.detach().cpu().long()
    X_test = X_test.detach().cpu()
    y_test = y_test.detach().cpu().long()

    if use_validation_split:
        labels = y_train_full.numpy()
        indices = np.arange(len(labels))

        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=None,
            )

        train_ds = TensorDataset(X_train_full[train_idx], y_train_full[train_idx])
        val_ds = TensorDataset(X_train_full[val_idx], y_train_full[val_idx])
    else:
        train_ds = TensorDataset(X_train_full, y_train_full)
        val_ds = TensorDataset(X_test, y_test)

    test_ds = TensorDataset(X_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# Models
# =============================================================================

class PlainSSMClassifier(nn.Module):
    """Plain state-space classifier with uniform average pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        z_list = []
        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(1)
            h = h @ self.A.T + x_t @ self.B.T
            z_t = torch.tanh(self.feature_layer(torch.cat([h, x_t], dim=1)))
            z_list.append(z_t)

        z = torch.stack(z_list, dim=1)
        u = z.mean(dim=1)
        logits = self.classifier(u)

        return logits, {"z": z, "pooling_type": "mean"}


class GatedSSMClassifier(nn.Module):
    """Gated state-space classifier with normalized or unnormalized pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_classes: int,
        pooling_type: str = "unnormalized",
        eps: float = 1e-8,
    ):
        super().__init__()

        if pooling_type not in {"unnormalized", "normalized"}:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")

        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.eps = eps

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        z_list = []
        alpha_list = []

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(1)
            h = h @ self.A.T + x_t @ self.B.T

            z_t = torch.tanh(self.feature_layer(torch.cat([h, x_t], dim=1)))
            alpha_t = torch.sigmoid(self.gate_layer(z_t))

            z_list.append(z_t)
            alpha_list.append(alpha_t)

        z = torch.stack(z_list, dim=1)          # [batch, T, latent_dim]
        alpha = torch.stack(alpha_list, dim=1)  # [batch, T, 1]
        evidence = alpha * z

        if self.pooling_type == "unnormalized":
            u = evidence.sum(dim=1)
        else:
            gate_mass = alpha.sum(dim=1) + self.eps
            u = evidence.sum(dim=1) / gate_mass

        logits = self.classifier(u)

        return logits, {
            "z": z,
            "alpha": alpha.squeeze(-1),
            "A": self.A,
            "pooling_type": self.pooling_type,
        }


# =============================================================================
# Training and evaluation
# =============================================================================

def evaluate_model(model: nn.Module, loader: DataLoader, run_device: torch.device) -> Dict[str, float]:
    """Evaluate accuracy and macro-F1."""
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(run_device)
            y_batch = y_batch.to(run_device)

            logits, _ = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y_batch.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_gate_statistics(model: nn.Module, loader: DataLoader, run_device: torch.device) -> Dict[str, float]:
    """Compute descriptive statistics of gate values on the test set."""
    model.eval()
    alpha_values: List[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(run_device)
            _, aux = model(X_batch)

            if "alpha" not in aux:
                return {
                    "alpha_mean": "",
                    "alpha_std": "",
                    "alpha_min": "",
                    "alpha_max": "",
                }

            alpha_values.append(aux["alpha"].detach().cpu().reshape(-1))

    alpha = torch.cat(alpha_values)

    return {
        "alpha_mean": float(alpha.mean()),
        "alpha_std": float(alpha.std()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
    }


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute standard cross-entropy loss."""
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    return ce_loss, {"ce_loss": float(ce_loss.detach().cpu())}


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    n_classes: int,
    pooling_type: str,
    run_device: torch.device,
) -> nn.Module:
    """Instantiate one model variant."""
    if model_type == "plain":
        return PlainSSMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_classes=n_classes,
        ).to(run_device)

    if model_type == "gated":
        return GatedSSMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_classes=n_classes,
            pooling_type=pooling_type,
        ).to(run_device)

    raise ValueError(f"Unknown model_type: {model_type}")


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    dataset_name: str,
    model_name: str,
    model_type: str,
    pooling_type: str,
    seed: int,
) -> Dict:
    """Train one model and return a detail-row dictionary."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_metric = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    best_val_metrics = {"accuracy": np.nan, "macro_f1": np.nan}
    patience_counter = 0
    start_time = time.time()
    last_train_stats: Dict[str, float] = {}
    epochs_run = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses: List[float] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss, train_stats = compute_loss(logits, y_batch)
            last_train_stats = train_stats

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite loss detected in model {model_name} "
                    f"on {dataset_name}, seed={seed}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_max_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(model, val_loader, cfg.device)
        val_metric = val_metrics["macro_f1"]
        epochs_run = epoch + 1

        if val_metric > best_metric:
            best_metric = val_metric
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{dataset_name}][{model_name}][seed={seed}] "
            f"Epoch {epoch + 1:03d} | "
            f"TrainLoss={np.mean(epoch_losses):.4f} | "
            f"ValAcc={val_metrics['accuracy']:.4f} | "
            f"ValMacroF1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= cfg.patience:
            print(f"[{dataset_name}][{model_name}][seed={seed}] Early stopping triggered.")
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    inference_start = time.time()
    final_metrics = evaluate_model(model, test_loader, cfg.device)
    inference_time = time.time() - inference_start

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "model_type": model_type,
        "pooling_type": pooling_type,
        "seed": seed,
        "status": "success",
        "error_message": "",
        "accuracy": final_metrics["accuracy"],
        "macro_f1": final_metrics["macro_f1"],
        "val_accuracy": best_val_metrics["accuracy"],
        "val_macro_f1": best_val_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "n_params": count_parameters(model),
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "clip_grad_max_norm": cfg.clip_grad_max_norm,
    }

    result.update(last_train_stats)

    if hasattr(model, "gate_layer"):
        result.update(get_gate_statistics(model, test_loader, cfg.device))
    else:
        result.update({
            "alpha_mean": "",
            "alpha_std": "",
            "alpha_min": "",
            "alpha_max": "",
        })

    return result


def make_error_result(
    dataset_name: str,
    model_name: str,
    model_type: str,
    pooling_type: str,
    cfg: TrainConfig,
    seed: int,
    error_message: str,
) -> Dict:
    """Create a detail-row dictionary for failed runs."""
    return {
        "dataset": dataset_name,
        "model": model_name,
        "model_type": model_type,
        "pooling_type": pooling_type,
        "seed": seed,
        "status": "failed",
        "error_message": error_message,
        "accuracy": "",
        "macro_f1": "",
        "val_accuracy": "",
        "val_macro_f1": "",
        "best_epoch": "",
        "epochs_run": "",
        "train_time_sec": "",
        "inference_time_sec": "",
        "n_params": "",
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "clip_grad_max_norm": cfg.clip_grad_max_norm,
        "ce_loss": "",
        "alpha_mean": "",
        "alpha_std": "",
        "alpha_min": "",
        "alpha_max": "",
    }


def run_one_dataset_seed(dataset_name: str, seed: int, cfg: TrainConfig) -> None:
    """Run all aggregation variants for one dataset/seed pair."""
    print("\n" + "=" * 80)
    print(f"Running dataset={dataset_name}, seed={seed}")
    print("=" * 80)

    set_global_seed(seed)

    X_train_full, y_train_oh, X_test, y_test_oh = load_dataset(dataset_name)
    y_train_full = get_integer_labels_from_onehot(y_train_oh)
    y_test = get_integer_labels_from_onehot(y_test_oh)

    n_classes = int(torch.max(y_train_full).item() + 1)
    input_dim = 1

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        batch_size=cfg.batch_size,
        seed=seed,
        val_size=cfg.val_size,
        use_validation_split=cfg.use_validation_split,
    )

    for model_cfg in MODEL_CONFIGS:
        model_name = model_cfg["model_name"]
        model_type = model_cfg["model_type"]
        pooling_type = model_cfg["pooling_type"]

        try:
            set_global_seed(seed)

            model = build_model(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                latent_dim=cfg.latent_dim,
                n_classes=n_classes,
                pooling_type=pooling_type,
                run_device=cfg.device,
            )

            result = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                cfg=cfg,
                dataset_name=dataset_name,
                model_name=model_name,
                model_type=model_type,
                pooling_type=pooling_type,
                seed=seed,
            )

            append_row_to_csv(cfg.detail_csv, result, DETAIL_COLUMNS)
            print("Saved:", result)

        except Exception as exc:
            error_result = make_error_result(
                dataset_name=dataset_name,
                model_name=model_name,
                model_type=model_type,
                pooling_type=pooling_type,
                cfg=cfg,
                seed=seed,
                error_message=str(exc),
            )
            append_row_to_csv(cfg.detail_csv, error_result, DETAIL_COLUMNS)
            print(f"[{dataset_name}][{model_name}][seed={seed}] FAILED: {exc}")


def summarize_results(detail_csv: str | Path, summary_csv: str | Path) -> None:
    """Create dataset/model summary statistics from the detail CSV."""
    rows: List[Dict[str, str]] = []

    with open(detail_csv, "r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["model_type"], row["pooling_type"])
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []

    for (dataset, model, model_type, pooling_type), group_rows in groups.items():
        success_rows = [r for r in group_rows if r["status"] == "success"]
        failed_rows = [r for r in group_rows if r["status"] != "success"]

        acc = np.array([to_float(r["accuracy"]) for r in success_rows], dtype=float)
        f1 = np.array([to_float(r["macro_f1"]) for r in success_rows], dtype=float)
        train_time = np.array([to_float(r["train_time_sec"]) for r in success_rows], dtype=float)
        inference_time = np.array([to_float(r["inference_time_sec"]) for r in success_rows], dtype=float)
        alpha_mean = np.array([to_float(r["alpha_mean"]) for r in success_rows], dtype=float)
        alpha_std = np.array([to_float(r["alpha_std"]) for r in success_rows], dtype=float)

        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "model_type": model_type,
            "pooling_type": pooling_type,
            "n_success": len(success_rows),
            "n_failed": len(failed_rows),
            "accuracy_mean": safe_mean(acc),
            "accuracy_std": safe_std(acc),
            "accuracy_min": np.nanmin(acc) if len(acc[~np.isnan(acc)]) else "",
            "accuracy_max": np.nanmax(acc) if len(acc[~np.isnan(acc)]) else "",
            "macro_f1_mean": safe_mean(f1),
            "macro_f1_std": safe_std(f1),
            "macro_f1_min": np.nanmin(f1) if len(f1[~np.isnan(f1)]) else "",
            "macro_f1_max": np.nanmax(f1) if len(f1[~np.isnan(f1)]) else "",
            "train_time_mean_sec": safe_mean(train_time),
            "inference_time_mean_sec": safe_mean(inference_time),
            "alpha_mean_mean": safe_mean(alpha_mean),
            "alpha_std_mean": safe_mean(alpha_std),
        })

    model_order = {
        "PlainSSM": 0,
        "NormGated": 1,
        "Proposed_Unnormalized_Base": 2,
    }
    summary_rows.sort(key=lambda row: (row["dataset"], model_order.get(row["model"], 99)))

    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})

    print("\nSummary saved to:", summary_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the UCR multi-seed aggregation ablation experiment for "
            "PlainSSM, NormGated, and Proposed Base."
        )
    )

    parser.add_argument("--results-dir", type=str, default="Results")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--clip-grad-max-norm", type=float, default=0.5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--no-validation-split", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        dataset_names=list(args.datasets),
        seeds=list(args.seeds),
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        clip_grad_max_norm=args.clip_grad_max_norm,
        val_size=args.val_size,
        use_validation_split=not args.no_validation_split,
        results_dir=Path(args.results_dir),
        device=resolve_device(args.device),
    )

    cfg.detail_csv, cfg.summary_csv = make_results_csv_names(cfg.results_dir)

    print("RUNNING MULTI-SEED AGGREGATION ABLATION EXPERIMENT")
    print("Datasets:", cfg.dataset_names)
    print("Seeds:", cfg.seeds)
    print("Learning rate:", cfg.lr)
    print("Max epochs:", cfg.max_epochs)
    print("Patience:", cfg.patience)
    print("Validation split enabled:", cfg.use_validation_split)
    print("Validation size:", cfg.val_size)
    print("Gradient clipping max_norm:", cfg.clip_grad_max_norm)
    print("Device:", cfg.device)
    print("Detail results:", cfg.detail_csv)
    print("Summary results:", cfg.summary_csv)

    print("\nModel configurations:")
    for model_cfg in MODEL_CONFIGS:
        print(
            f"  {model_cfg['model_name']}: "
            f"model_type={model_cfg['model_type']}, "
            f"pooling_type={model_cfg['pooling_type']}"
        )

    for dataset_name in cfg.dataset_names:
        for seed in cfg.seeds:
            run_one_dataset_seed(dataset_name, seed, cfg)

    summarize_results(cfg.detail_csv, cfg.summary_csv)

    print("\nAggregation ablation experiment finished.")
    print("Detail results saved to:", cfg.detail_csv)
    print("Summary results saved to:", cfg.summary_csv)


if __name__ == "__main__":
    main()
