"""
Classifier-aware evidence deletion/insertion experiment for UCR datasets.

This script reproduces the score-comparison experiment used for the real-data
evidence analysis. It trains the proposed unnormalized evidence-gated SSM and
evaluates deletion/insertion at a fixed evidence ratio using six temporal
ranking scores:

    - random
    - latent_norm:   ||z_t||_2
    - gate_alpha:    alpha_t
    - evidence_norm: ||alpha_t z_t||_2
    - class_logit:   w_c_hat^T e_t
    - margin_logit:  (w_c_hat - w_c_prime)^T e_t

Outputs:
    - detail CSV: one row per dataset/seed/score/operation
    - summary CSV: dataset-level means over seeds
    - overall CSV: aggregate means over all dataset-seed rows

Expected companion file:
    process_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from process_datasets import (
    device as DEFAULT_DEVICE,
    get_integer_labels_from_onehot,
    load_dataset,
)


# ---------------------------------------------------------------------
# Default experiment configuration
# ---------------------------------------------------------------------

DEFAULT_DATASETS: Tuple[str, ...] = (
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
)

DEFAULT_SEEDS: Tuple[int, ...] = (2025, 2026, 2027, 2028, 2029)

SCORE_BASES: Tuple[str, ...] = (
    "random",
    "latent_norm",
    "gate_alpha",
    "evidence_norm",
    "class_logit",
    "margin_logit",
)

OPERATIONS: Tuple[str, ...] = ("deletion", "insertion")

DETAIL_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "model",
    "seed",
    "score_basis",
    "operation",
    "ratio",
    "status",
    "error_message",
    "accuracy",
    "macro_f1",
    "accuracy_drop",
    "macro_f1_drop",
    "original_accuracy",
    "original_macro_f1",
    "best_epoch",
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
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
    "evidence_norm_mean",
    "evidence_norm_std",
    "evidence_norm_min",
    "evidence_norm_max",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "model",
    "score_basis",
    "operation",
    "ratio",
    "n_success",
    "n_failed",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "accuracy_drop_mean",
    "accuracy_drop_std",
    "macro_f1_drop_mean",
    "macro_f1_drop_std",
    "failed_seeds",
)

OVERALL_COLUMNS: Tuple[str, ...] = (
    "score_basis",
    "operation",
    "ratio",
    "n_rows",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "accuracy_drop_mean",
    "accuracy_drop_std",
    "macro_f1_drop_mean",
    "macro_f1_drop_std",
)


@dataclass
class TrainConfig:
    dataset_names: List[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))

    model_name: str = "Proposed_Unnormalized_Base"

    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 50
    patience: int = 10
    hidden_dim: int = 64
    latent_dim: int = 64
    clip_grad_max_norm: float = 0.5

    use_validation_split: bool = True
    val_size: float = 0.2

    ratios: List[float] = field(default_factory=lambda: [0.10])
    score_bases: List[str] = field(default_factory=lambda: list(SCORE_BASES))
    operations: List[str] = field(default_factory=lambda: list(OPERATIONS))

    random_repeats: int = 3
    results_dir: Path = Path("Results")
    device: torch.device = DEFAULT_DEVICE

    detail_csv: Path = Path()
    summary_csv: Path = Path()
    overall_csv: Path = Path()


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def parse_csv_list(value: str, cast=str) -> List:
    """Parse comma-separated command-line lists."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [cast(item) for item in items]


def resolve_device(value: str) -> torch.device:
    """Resolve a command-line device string."""
    if value == "auto":
        return DEFAULT_DEVICE
    return torch.device(value)


def set_global_seed(seed: int) -> None:
    """Set all random seeds used by this experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_results_file_names(results_dir: Path) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_csv = results_dir / f"ucr_score_deletion_insertion_detail_{timestamp}.csv"
    summary_csv = results_dir / f"ucr_score_deletion_insertion_summary_{timestamp}.csv"
    overall_csv = results_dir / f"ucr_score_deletion_insertion_overall_{timestamp}.csv"
    return detail_csv, summary_csv, overall_csv


def append_row(csv_path: Path, row: Dict, columns: Sequence[str]) -> None:
    """Append one row to a CSV file, creating the header if necessary."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    full_row = {col: row.get(col, "") for col in columns}

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


def safe_std(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def safe_mean(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def to_float(value) -> float:
    try:
        if value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def move_batch_to_device(batch: Tuple[torch.Tensor, torch.Tensor], target_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x_batch, y_batch = batch
    return x_batch.to(target_device), y_batch.to(target_device)


def create_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    seed: int,
    use_validation_split: bool,
    val_size: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test DataLoaders."""
    if use_validation_split:
        indices = np.arange(len(x_train))
        y_np = y_train.detach().cpu().numpy()

        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=y_np,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=None,
            )

        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        x_val, y_val = x_train[val_idx], y_train[val_idx]
    else:
        x_tr, y_tr = x_train, y_train
        x_val, y_val = x_test, y_test

    generator = torch.Generator()
    generator.manual_seed(seed)

    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class ProposedUnnormalizedSSM(nn.Module):
    """Simple evidence-gated SSM classifier with unnormalized accumulation."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a univariate time series.

        Args:
            x: Tensor of shape (batch_size, seq_len).

        Returns:
            z: Tensor of shape (batch_size, seq_len, latent_dim).
            alpha: Tensor of shape (batch_size, seq_len).
        """
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch_size, seq_len), got {tuple(x.shape)}.")

        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        z_list: List[torch.Tensor] = []
        alpha_list: List[torch.Tensor] = []

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(1)
            h = h @ self.A.T + x_t @ self.B.T
            z_t = torch.tanh(self.feature_layer(torch.cat([h, x_t], dim=1)))
            alpha_t = torch.sigmoid(self.gate_layer(z_t))

            z_list.append(z_t)
            alpha_list.append(alpha_t)

        z = torch.stack(z_list, dim=1)
        alpha = torch.stack(alpha_list, dim=1).squeeze(-1)

        return z, alpha

    def classify_from_evidence(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Compute logits from u = sum_t alpha_t z_t."""
        u = (alpha.unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def classify_from_masked_evidence(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits after deletion/insertion at the evidence level.

        keep_mask shape: (batch_size, seq_len), with 1 for retained terms.
        """
        u = ((alpha * keep_mask).unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z, alpha = self.encode(x)
        logits = self.classify_from_evidence(z, alpha)
        return logits, {"z": z, "alpha": alpha, "A": self.A}


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def evaluate_loader(model: nn.Module, loader: DataLoader, target_device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch = move_batch_to_device(batch, target_device)
            logits, _ = model(x_batch)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y_batch.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    dataset_name: str,
    seed: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_metric = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            x_batch, y_batch = move_batch_to_device(batch, cfg.device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = loss_fn(logits, y_batch)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected on {dataset_name}, seed={seed}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_max_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader, cfg.device)
        val_metric = val_metrics["macro_f1"]

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{dataset_name}][{cfg.model_name}][seed={seed}] "
            f"Epoch {epoch + 1:03d} | "
            f"TrainLoss={np.mean(epoch_losses):.4f} | "
            f"ValAcc={val_metrics['accuracy']:.4f} | "
            f"ValMacroF1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= cfg.patience:
            print(f"[{dataset_name}][{cfg.model_name}][seed={seed}] Early stopping triggered.")
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(cfg.device)

    return model, {"best_epoch": best_epoch, "train_time_sec": train_time}


def compute_evidence_statistics(
    model: ProposedUnnormalizedSSM,
    loader: DataLoader,
    target_device: torch.device,
) -> Dict[str, float]:
    model.eval()
    alpha_values: List[torch.Tensor] = []
    evidence_norm_values: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            x_batch, _ = move_batch_to_device(batch, target_device)
            z, alpha = model.encode(x_batch)
            evidence = alpha.unsqueeze(-1) * z
            evidence_norm = torch.linalg.vector_norm(evidence, ord=2, dim=-1)

            alpha_values.append(alpha.detach().cpu().reshape(-1))
            evidence_norm_values.append(evidence_norm.detach().cpu().reshape(-1))

    alpha_all = torch.cat(alpha_values)
    evidence_norm_all = torch.cat(evidence_norm_values)

    return {
        "alpha_mean": float(alpha_all.mean()),
        "alpha_std": float(alpha_all.std()),
        "alpha_min": float(alpha_all.min()),
        "alpha_max": float(alpha_all.max()),
        "evidence_norm_mean": float(evidence_norm_all.mean()),
        "evidence_norm_std": float(evidence_norm_all.std()),
        "evidence_norm_min": float(evidence_norm_all.min()),
        "evidence_norm_max": float(evidence_norm_all.max()),
    }


# ---------------------------------------------------------------------
# Evidence scores and operations
# ---------------------------------------------------------------------

def compute_scores(
    z: torch.Tensor,
    alpha: torch.Tensor,
    logits: torch.Tensor,
    model: ProposedUnnormalizedSSM,
    score_basis: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Compute temporal ranking scores for deletion/insertion."""
    batch_size, seq_len, _ = z.shape
    evidence = alpha.unsqueeze(-1) * z

    if score_basis == "random":
        return torch.rand(batch_size, seq_len, device=z.device, generator=generator)

    if score_basis == "latent_norm":
        return torch.linalg.vector_norm(z, ord=2, dim=-1)

    if score_basis == "gate_alpha":
        return alpha

    if score_basis == "evidence_norm":
        return torch.linalg.vector_norm(evidence, ord=2, dim=-1)

    classifier_weights = model.classifier.weight
    pred_class = torch.argmax(logits, dim=1)
    w_pred = classifier_weights[pred_class]

    if score_basis == "class_logit":
        return torch.sum(evidence * w_pred[:, None, :], dim=-1)

    if score_basis == "margin_logit":
        logits_masked = logits.clone()
        row_idx = torch.arange(batch_size, device=logits.device)
        logits_masked[row_idx, pred_class] = -1e9

        competitor = torch.argmax(logits_masked, dim=1)
        w_comp = classifier_weights[competitor]

        return torch.sum(evidence * (w_pred - w_comp)[:, None, :], dim=-1)

    raise ValueError(f"Unknown score_basis: {score_basis}")


def make_mask_from_scores(scores: torch.Tensor, operation: str, ratio: float) -> torch.Tensor:
    """
    Create an evidence keep mask.

    For deletion, top-ranked terms are removed.
    For insertion, only top-ranked terms are retained.
    """
    batch_size, seq_len = scores.shape
    k = int(round(ratio * seq_len))
    k = max(1, min(k, seq_len))

    top_idx = torch.topk(scores, k=k, dim=1, largest=True).indices

    if operation == "deletion":
        mask = torch.ones(batch_size, seq_len, device=scores.device)
        mask.scatter_(dim=1, index=top_idx, value=0.0)
        return mask

    if operation == "insertion":
        mask = torch.zeros(batch_size, seq_len, device=scores.device)
        mask.scatter_(dim=1, index=top_idx, value=1.0)
        return mask

    raise ValueError(f"Unknown operation: {operation}")


def evaluate_evidence_operation(
    model: ProposedUnnormalizedSSM,
    loader: DataLoader,
    score_basis: str,
    operation: str,
    ratio: float,
    seed: int,
    random_repeats: int,
    target_device: torch.device,
) -> Dict[str, float]:
    """Evaluate deletion or insertion at the representation level."""
    model.eval()
    repeats = random_repeats if score_basis == "random" else 1

    y_true: List[int] = []
    y_pred: List[int] = []
    total_inference_time = 0.0

    with torch.no_grad():
        for repeat_idx in range(repeats):
            generator = torch.Generator(device=target_device)
            generator.manual_seed(seed + 1000 * (repeat_idx + 1))

            for batch in loader:
                x_batch, y_batch = move_batch_to_device(batch, target_device)
                start_time = time.time()

                z, alpha = model.encode(x_batch)
                original_logits = model.classify_from_evidence(z, alpha)

                scores = compute_scores(
                    z=z,
                    alpha=alpha,
                    logits=original_logits,
                    model=model,
                    score_basis=score_basis,
                    generator=generator,
                )

                keep_mask = make_mask_from_scores(scores=scores, operation=operation, ratio=ratio)
                logits = model.classify_from_masked_evidence(z=z, alpha=alpha, keep_mask=keep_mask)
                preds = torch.argmax(logits, dim=1)

                total_inference_time += time.time() - start_time

                y_true.extend(y_batch.detach().cpu().numpy().tolist())
                y_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "inference_time_sec": total_inference_time,
    }


# ---------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------

def make_error_row(
    cfg: TrainConfig,
    dataset_name: str,
    seed: int,
    score_basis: str,
    operation: str,
    ratio: float,
    error_message: str,
) -> Dict:
    return {
        "dataset": dataset_name,
        "model": cfg.model_name,
        "seed": seed,
        "score_basis": score_basis,
        "operation": operation,
        "ratio": ratio,
        "status": "failed",
        "error_message": error_message,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "clip_grad_max_norm": cfg.clip_grad_max_norm,
    }


def write_result_row(
    cfg: TrainConfig,
    dataset_name: str,
    seed: int,
    score_basis: str,
    operation: str,
    ratio: float,
    eval_result: Dict[str, float],
    original_accuracy: float,
    original_macro_f1: float,
    train_info: Dict[str, float],
    n_params: int,
    evidence_stats: Dict[str, float],
) -> None:
    row = {
        "dataset": dataset_name,
        "model": cfg.model_name,
        "seed": seed,
        "score_basis": score_basis,
        "operation": operation,
        "ratio": ratio,
        "status": "success",
        "error_message": "",
        "accuracy": eval_result["accuracy"],
        "macro_f1": eval_result["macro_f1"],
        "accuracy_drop": original_accuracy - eval_result["accuracy"],
        "macro_f1_drop": original_macro_f1 - eval_result["macro_f1"],
        "original_accuracy": original_accuracy,
        "original_macro_f1": original_macro_f1,
        "best_epoch": train_info["best_epoch"],
        "train_time_sec": train_info["train_time_sec"],
        "inference_time_sec": eval_result["inference_time_sec"],
        "n_params": n_params,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "clip_grad_max_norm": cfg.clip_grad_max_norm,
    }
    row.update(evidence_stats)
    append_row(cfg.detail_csv, row, DETAIL_COLUMNS)


def run_one_dataset_seed(cfg: TrainConfig, dataset_name: str, seed: int) -> None:
    print("\n" + "=" * 80)
    print(f"UCR score deletion/insertion | dataset={dataset_name} | seed={seed}")
    print("=" * 80)

    set_global_seed(seed)

    try:
        x_train, y_train_onehot, x_test, y_test_onehot = load_dataset(dataset_name)

        y_train = get_integer_labels_from_onehot(y_train_onehot).long()
        y_test = get_integer_labels_from_onehot(y_test_onehot).long()

        n_classes = int(torch.max(y_train).item() + 1)
        input_dim = 1

        train_loader, val_loader, test_loader = create_dataloaders(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            batch_size=cfg.batch_size,
            seed=seed,
            use_validation_split=cfg.use_validation_split,
            val_size=cfg.val_size,
        )

        set_global_seed(seed)

        model = ProposedUnnormalizedSSM(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            n_classes=n_classes,
        ).to(cfg.device)

        model, train_info = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            dataset_name=dataset_name,
            seed=seed,
        )

        n_params = count_parameters(model)
        evidence_stats = compute_evidence_statistics(model, test_loader, cfg.device)

        original_eval = evaluate_loader(model, test_loader, cfg.device)
        original_accuracy = original_eval["accuracy"]
        original_macro_f1 = original_eval["macro_f1"]

        write_result_row(
            cfg=cfg,
            dataset_name=dataset_name,
            seed=seed,
            score_basis="none",
            operation="original",
            ratio=0.0,
            eval_result={
                "accuracy": original_accuracy,
                "macro_f1": original_macro_f1,
                "inference_time_sec": 0.0,
            },
            original_accuracy=original_accuracy,
            original_macro_f1=original_macro_f1,
            train_info=train_info,
            n_params=n_params,
            evidence_stats=evidence_stats,
        )

        print(
            f"[{dataset_name}][seed={seed}] original | "
            f"Acc={original_accuracy:.4f} | MacroF1={original_macro_f1:.4f}"
        )

        for ratio in cfg.ratios:
            for score_basis in cfg.score_bases:
                for operation in cfg.operations:
                    eval_result = evaluate_evidence_operation(
                        model=model,
                        loader=test_loader,
                        score_basis=score_basis,
                        operation=operation,
                        ratio=ratio,
                        seed=seed,
                        random_repeats=cfg.random_repeats,
                        target_device=cfg.device,
                    )

                    write_result_row(
                        cfg=cfg,
                        dataset_name=dataset_name,
                        seed=seed,
                        score_basis=score_basis,
                        operation=operation,
                        ratio=ratio,
                        eval_result=eval_result,
                        original_accuracy=original_accuracy,
                        original_macro_f1=original_macro_f1,
                        train_info=train_info,
                        n_params=n_params,
                        evidence_stats=evidence_stats,
                    )

                    print(
                        f"[{dataset_name}][seed={seed}] "
                        f"{score_basis} | {operation} {ratio:.0%} | "
                        f"Acc={eval_result['accuracy']:.4f} "
                        f"(drop={original_accuracy - eval_result['accuracy']:.4f}) | "
                        f"MacroF1={eval_result['macro_f1']:.4f} "
                        f"(drop={original_macro_f1 - eval_result['macro_f1']:.4f})"
                    )

    except Exception as exc:
        print(f"[{dataset_name}][seed={seed}] FAILED: {exc}")

        append_row(
            cfg.detail_csv,
            make_error_row(cfg, dataset_name, seed, "none", "original", 0.0, str(exc)),
            DETAIL_COLUMNS,
        )

        for ratio in cfg.ratios:
            for score_basis in cfg.score_bases:
                for operation in cfg.operations:
                    append_row(
                        cfg.detail_csv,
                        make_error_row(cfg, dataset_name, seed, score_basis, operation, ratio, str(exc)),
                        DETAIL_COLUMNS,
                    )


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_results(detail_csv: Path, summary_csv: Path, overall_csv: Path) -> None:
    with detail_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = {}

    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["score_basis"],
            row["operation"],
            row["ratio"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []

    for (dataset, model, score_basis, operation, ratio), group_rows in groups.items():
        success_rows = [row for row in group_rows if row["status"] == "success"]
        failed_rows = [row for row in group_rows if row["status"] != "success"]

        accuracy = np.array([to_float(row["accuracy"]) for row in success_rows], dtype=float)
        macro_f1 = np.array([to_float(row["macro_f1"]) for row in success_rows], dtype=float)
        accuracy_drop = np.array([to_float(row["accuracy_drop"]) for row in success_rows], dtype=float)
        macro_f1_drop = np.array([to_float(row["macro_f1_drop"]) for row in success_rows], dtype=float)

        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "score_basis": score_basis,
            "operation": operation,
            "ratio": ratio,
            "n_success": len(success_rows),
            "n_failed": len(failed_rows),
            "accuracy_mean": safe_mean(accuracy) if success_rows else "",
            "accuracy_std": safe_std(accuracy) if success_rows else "",
            "macro_f1_mean": safe_mean(macro_f1) if success_rows else "",
            "macro_f1_std": safe_std(macro_f1) if success_rows else "",
            "accuracy_drop_mean": safe_mean(accuracy_drop) if success_rows else "",
            "accuracy_drop_std": safe_std(accuracy_drop) if success_rows else "",
            "macro_f1_drop_mean": safe_mean(macro_f1_drop) if success_rows else "",
            "macro_f1_drop_std": safe_std(macro_f1_drop) if success_rows else "",
            "failed_seeds": ";".join([row["seed"] for row in failed_rows]),
        })

    score_order = {
        "none": 0,
        "random": 1,
        "latent_norm": 2,
        "gate_alpha": 3,
        "evidence_norm": 4,
        "class_logit": 5,
        "margin_logit": 6,
    }
    operation_order = {"original": 0, "deletion": 1, "insertion": 2}

    summary_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            float(row["ratio"]),
            operation_order.get(str(row["operation"]), 99),
            score_order.get(str(row["score_basis"]), 99),
        )
    )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(summary_rows)

    success_rows = [
        row
        for row in rows
        if row["status"] == "success" and row["operation"] in {"deletion", "insertion"}
    ]

    overall_groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}

    for row in success_rows:
        key = (row["score_basis"], row["operation"], row["ratio"])
        overall_groups.setdefault(key, []).append(row)

    overall_rows: List[Dict] = []

    for (score_basis, operation, ratio), group_rows in overall_groups.items():
        accuracy = np.array([to_float(row["accuracy"]) for row in group_rows], dtype=float)
        macro_f1 = np.array([to_float(row["macro_f1"]) for row in group_rows], dtype=float)
        accuracy_drop = np.array([to_float(row["accuracy_drop"]) for row in group_rows], dtype=float)
        macro_f1_drop = np.array([to_float(row["macro_f1_drop"]) for row in group_rows], dtype=float)

        overall_rows.append({
            "score_basis": score_basis,
            "operation": operation,
            "ratio": ratio,
            "n_rows": len(group_rows),
            "accuracy_mean": safe_mean(accuracy),
            "accuracy_std": safe_std(accuracy),
            "macro_f1_mean": safe_mean(macro_f1),
            "macro_f1_std": safe_std(macro_f1),
            "accuracy_drop_mean": safe_mean(accuracy_drop),
            "accuracy_drop_std": safe_std(accuracy_drop),
            "macro_f1_drop_mean": safe_mean(macro_f1_drop),
            "macro_f1_drop_std": safe_std(macro_f1_drop),
        })

    overall_rows.sort(
        key=lambda row: (
            float(row["ratio"]),
            operation_order.get(str(row["operation"]), 99),
            score_order.get(str(row["score_basis"]), 99),
        )
    )

    with overall_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OVERALL_COLUMNS))
        writer.writeheader()
        writer.writerows(overall_rows)


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UCR classifier-aware evidence deletion/insertion experiments."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated UCR dataset names.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated integer random seeds.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="Results",
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--clip-grad-max-norm", type=float, default=0.5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--no-validation-split",
        action="store_true",
        help="Use the test split as validation. Intended only for exploratory checks.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.10",
        help="Comma-separated evidence ratios, e.g. 0.10 or 0.10,0.20.",
    )
    parser.add_argument("--random-repeats", type=int, default=3)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        dataset_names=parse_csv_list(args.datasets, str),
        seeds=parse_csv_list(args.seeds, int),
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        clip_grad_max_norm=args.clip_grad_max_norm,
        use_validation_split=not args.no_validation_split,
        val_size=args.val_size,
        ratios=parse_csv_list(args.ratios, float),
        random_repeats=args.random_repeats,
        results_dir=results_dir,
        device=resolve_device(args.device),
    )

    cfg.detail_csv, cfg.summary_csv, cfg.overall_csv = make_results_file_names(results_dir)

    return cfg


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    print("RUNNING UCR SCORE DELETION/INSERTION EXPERIMENT")
    print("Datasets:", cfg.dataset_names)
    print("Seeds:", cfg.seeds)
    print("Model:", cfg.model_name)
    print("Score bases:", cfg.score_bases)
    print("Operations:", cfg.operations)
    print("Ratios:", cfg.ratios)
    print("Random repeats:", cfg.random_repeats)
    print("Validation split:", cfg.use_validation_split, "| val_size:", cfg.val_size)
    print("Device:", cfg.device)
    print("Detail CSV:", cfg.detail_csv)
    print("Summary CSV:", cfg.summary_csv)
    print("Overall CSV:", cfg.overall_csv)

    for dataset_name in cfg.dataset_names:
        for seed in cfg.seeds:
            run_one_dataset_seed(cfg, dataset_name, seed)

    summarize_results(cfg.detail_csv, cfg.summary_csv, cfg.overall_csv)

    print("\nUCR score deletion/insertion experiment finished.")
    print("Detail results saved to:", cfg.detail_csv)
    print("Summary results saved to:", cfg.summary_csv)
    print("Overall results saved to:", cfg.overall_csv)


if __name__ == "__main__":
    main()
