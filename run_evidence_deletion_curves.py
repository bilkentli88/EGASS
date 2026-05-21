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

from process_datasets import (
    device as DEFAULT_DEVICE,
    get_integer_labels_from_onehot,
    load_dataset,
)


DETAIL_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "score_basis",
    "deletion_type",
    "deletion_ratio",
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
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "score_basis",
    "deletion_type",
    "deletion_ratio",
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
]

TEN_UCR_DATASETS = [
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


@dataclass
class TrainConfig:
    """Configuration for the evidence-norm deletion-curve experiment."""

    dataset_names: List[str] = field(default_factory=lambda: TEN_UCR_DATASETS.copy())
    seeds: List[int] = field(default_factory=lambda: [2025, 2026, 2027, 2028, 2029])

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

    deletion_ratios: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50]
    )
    score_bases: List[str] = field(default_factory=lambda: ["evidence_norm"])
    deletion_types: List[str] = field(
        default_factory=lambda: ["original", "high_score", "low_score", "random"]
    )
    random_deletion_repeats: int = 3

    results_dir: Path = Path("Results")
    device: torch.device | str = DEFAULT_DEVICE

    detail_csv: str = ""
    summary_csv: str = ""
    average_curve_pdf: str = ""
    average_curve_png: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run evidence-level deletion curves for the proposed unnormalized "
            "evidence-gated SSM on the ten selected UCR datasets."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="Results",
        help="Directory where CSV summaries and figures will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. 'cpu' or 'cuda'. By default uses process_datasets.device.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset list. Defaults to the ten selected UCR datasets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed list. Defaults to 2025 2026 2027 2028 2029.",
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--random-deletion-repeats", type=int, default=3)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_results_file_names(results_dir: Path) -> Tuple[str, str, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_csv = results_dir / f"evidence_deletion_curves_detail_{timestamp}.csv"
    summary_csv = results_dir / f"evidence_deletion_curves_summary_{timestamp}.csv"
    average_curve_pdf = results_dir / f"evidence_deletion_curves_average_{timestamp}.pdf"
    average_curve_png = results_dir / f"evidence_deletion_curves_average_{timestamp}.png"
    return str(detail_csv), str(summary_csv), str(average_curve_pdf), str(average_curve_png)


def append_row(csv_path: str | Path, row: Dict, columns: List[str]) -> None:
    csv_path = Path(csv_path)
    exists = csv_path.exists()
    full_row = {col: row.get(col, "") for col in columns}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


def _move_tensor_dataset_to_device(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    run_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        X_train.to(run_device),
        y_train.to(run_device),
        X_test.to(run_device),
        y_test.to(run_device),
    )


def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    seed: int,
    use_validation_split: bool,
    val_size: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if use_validation_split:
        indices = np.arange(len(X_train))
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

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
    else:
        X_tr = X_train
        y_tr = y_train
        X_val = X_test
        y_val = y_test

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class ProposedUnnormalizedSSM(nn.Module):
    """Simple evidence-gated state-space classifier used in the paper."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        z = torch.stack(z_list, dim=1)
        alpha = torch.stack(alpha_list, dim=1).squeeze(-1)
        return z, alpha

    def classify_from_evidence(self, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        u = (alpha.unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def classify_from_masked_evidence(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        u = ((alpha * keep_mask).unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def forward(self, x: torch.Tensor):
        z, alpha = self.encode(x)
        logits = self.classify_from_evidence(z, alpha)
        return logits, {"z": z, "alpha": alpha, "A": self.A}


def evaluate_loader(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits, _ = model(X_batch)
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

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits, _ = model(X_batch)
            loss = loss_fn(logits, y_batch)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected on {dataset_name}, seed {seed}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_max_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader)
        val_metric = val_metrics["macro_f1"]

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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

    return model, {"best_epoch": best_epoch, "train_time_sec": train_time}


def compute_evidence_statistics(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    alpha_values: List[torch.Tensor] = []
    evidence_norm_values: List[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, _ in loader:
            z, alpha = model.encode(X_batch)
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


def make_keep_mask_from_scores(
    scores: torch.Tensor,
    deletion_type: str,
    deletion_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    batch_size, seq_len = scores.shape
    keep_mask = torch.ones(batch_size, seq_len, device=scores.device)

    if deletion_type == "original" or deletion_ratio <= 0:
        return keep_mask

    k = int(round(deletion_ratio * seq_len))
    k = max(1, min(k, seq_len))

    if deletion_type == "high_score":
        delete_idx = torch.topk(scores, k=k, dim=1, largest=True).indices
    elif deletion_type == "low_score":
        delete_idx = torch.topk(scores, k=k, dim=1, largest=False).indices
    elif deletion_type == "random":
        if generator is None:
            raise ValueError("A generator must be provided for random deletion.")
        random_indices = []
        for _ in range(batch_size):
            perm = torch.randperm(seq_len, device=scores.device, generator=generator)
            random_indices.append(perm[:k])
        delete_idx = torch.stack(random_indices, dim=0)
    else:
        raise ValueError(f"Unknown deletion_type: {deletion_type}")

    keep_mask.scatter_(dim=1, index=delete_idx, value=0.0)
    return keep_mask


def compute_scores(z: torch.Tensor, alpha: torch.Tensor, score_basis: str) -> torch.Tensor:
    if score_basis == "evidence_norm":
        evidence = alpha.unsqueeze(-1) * z
        return torch.linalg.vector_norm(evidence, ord=2, dim=-1)

    raise ValueError(f"Unknown score_basis: {score_basis}")


def evaluate_evidence_deletion(
    model: ProposedUnnormalizedSSM,
    loader: DataLoader,
    score_basis: str,
    deletion_type: str,
    deletion_ratio: float,
    seed: int,
    random_repeats: int,
    run_device: torch.device,
) -> Dict[str, float]:
    model.eval()
    repeats = random_repeats if deletion_type == "random" and deletion_ratio > 0 else 1

    all_true: List[int] = []
    all_pred: List[int] = []
    total_inference_time = 0.0

    with torch.no_grad():
        for repeat_idx in range(repeats):
            generator = torch.Generator(device=run_device)
            generator.manual_seed(seed + 1000 * (repeat_idx + 1))

            for X_batch, y_batch in loader:
                start = time.time()

                z, alpha = model.encode(X_batch)
                scores = compute_scores(z, alpha, score_basis)
                keep_mask = make_keep_mask_from_scores(
                    scores=scores,
                    deletion_type=deletion_type,
                    deletion_ratio=deletion_ratio,
                    generator=generator,
                )

                logits = model.classify_from_masked_evidence(
                    z=z,
                    alpha=alpha,
                    keep_mask=keep_mask,
                )
                preds = torch.argmax(logits, dim=1)

                total_inference_time += time.time() - start

                all_true.extend(y_batch.detach().cpu().numpy().tolist())
                all_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "inference_time_sec": total_inference_time,
    }


def make_error_row(
    cfg: TrainConfig,
    dataset_name: str,
    seed: int,
    score_basis: str,
    deletion_type: str,
    deletion_ratio: float,
    error_message: str,
) -> Dict:
    return {
        "dataset": dataset_name,
        "model": cfg.model_name,
        "seed": seed,
        "score_basis": score_basis,
        "deletion_type": deletion_type,
        "deletion_ratio": deletion_ratio,
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
    deletion_type: str,
    deletion_ratio: float,
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
        "deletion_type": deletion_type,
        "deletion_ratio": deletion_ratio,
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
    print(f"Evidence-level deletion curves | dataset={dataset_name} | seed={seed}")
    print("=" * 80)

    set_global_seed(seed)
    run_device = torch.device(cfg.device)

    try:
        X_train, y_train_oh, X_test, y_test_oh = load_dataset(dataset_name)
        y_train = get_integer_labels_from_onehot(y_train_oh)
        y_test = get_integer_labels_from_onehot(y_test_oh)

        X_train, y_train, X_test, y_test = _move_tensor_dataset_to_device(
            X_train, y_train, X_test, y_test, run_device
        )

        n_classes = int(torch.max(y_train).item() + 1)
        input_dim = 1

        train_loader, val_loader, test_loader = create_dataloaders(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
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
        ).to(run_device)

        model, train_info = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            dataset_name=dataset_name,
            seed=seed,
        )

        n_params = count_parameters(model)
        evidence_stats = compute_evidence_statistics(model, test_loader)

        original_eval = evaluate_evidence_deletion(
            model=model,
            loader=test_loader,
            score_basis="evidence_norm",
            deletion_type="original",
            deletion_ratio=0.0,
            seed=seed,
            random_repeats=1,
            run_device=run_device,
        )

        original_accuracy = original_eval["accuracy"]
        original_macro_f1 = original_eval["macro_f1"]

        write_result_row(
            cfg=cfg,
            dataset_name=dataset_name,
            seed=seed,
            score_basis="none",
            deletion_type="original",
            deletion_ratio=0.0,
            eval_result=original_eval,
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

        for score_basis in cfg.score_bases:
            for deletion_ratio in cfg.deletion_ratios:
                for deletion_type in ["high_score", "low_score", "random"]:
                    deletion_eval = evaluate_evidence_deletion(
                        model=model,
                        loader=test_loader,
                        score_basis=score_basis,
                        deletion_type=deletion_type,
                        deletion_ratio=deletion_ratio,
                        seed=seed,
                        random_repeats=cfg.random_deletion_repeats,
                        run_device=run_device,
                    )

                    write_result_row(
                        cfg=cfg,
                        dataset_name=dataset_name,
                        seed=seed,
                        score_basis=score_basis,
                        deletion_type=deletion_type,
                        deletion_ratio=deletion_ratio,
                        eval_result=deletion_eval,
                        original_accuracy=original_accuracy,
                        original_macro_f1=original_macro_f1,
                        train_info=train_info,
                        n_params=n_params,
                        evidence_stats=evidence_stats,
                    )

                    print(
                        f"[{dataset_name}][seed={seed}] "
                        f"{score_basis} | {deletion_type} {deletion_ratio:.0%} | "
                        f"Acc={deletion_eval['accuracy']:.4f} "
                        f"(drop={original_accuracy - deletion_eval['accuracy']:.4f}) | "
                        f"MacroF1={deletion_eval['macro_f1']:.4f} "
                        f"(drop={original_macro_f1 - deletion_eval['macro_f1']:.4f})"
                    )

    except Exception as exc:
        print(f"[{dataset_name}][seed={seed}] FAILED: {exc}")

        append_row(
            cfg.detail_csv,
            make_error_row(cfg, dataset_name, seed, "none", "original", 0.0, str(exc)),
            DETAIL_COLUMNS,
        )

        for score_basis in cfg.score_bases:
            for deletion_ratio in cfg.deletion_ratios:
                for deletion_type in ["high_score", "low_score", "random"]:
                    append_row(
                        cfg.detail_csv,
                        make_error_row(
                            cfg,
                            dataset_name,
                            seed,
                            score_basis,
                            deletion_type,
                            deletion_ratio,
                            str(exc),
                        ),
                        DETAIL_COLUMNS,
                    )


def summarize_results(detail_csv: str | Path, summary_csv: str | Path) -> None:
    detail_csv = Path(detail_csv)
    rows: List[Dict[str, str]] = []

    with open(detail_csv, "r", newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["score_basis"],
            row["deletion_type"],
            row["deletion_ratio"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (dataset, model, score_basis, deletion_type, deletion_ratio), group_rows in groups.items():
        success_rows = [r for r in group_rows if r["status"] == "success"]
        failed_rows = [r for r in group_rows if r["status"] != "success"]

        acc = np.array([to_float(r["accuracy"]) for r in success_rows], dtype=float)
        f1 = np.array([to_float(r["macro_f1"]) for r in success_rows], dtype=float)
        acc_drop = np.array([to_float(r["accuracy_drop"]) for r in success_rows], dtype=float)
        f1_drop = np.array([to_float(r["macro_f1_drop"]) for r in success_rows], dtype=float)

        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "score_basis": score_basis,
                "deletion_type": deletion_type,
                "deletion_ratio": deletion_ratio,
                "n_success": len(success_rows),
                "n_failed": len(failed_rows),
                "accuracy_mean": safe_mean(acc) if success_rows else "",
                "accuracy_std": safe_std(acc) if success_rows else "",
                "macro_f1_mean": safe_mean(f1) if success_rows else "",
                "macro_f1_std": safe_std(f1) if success_rows else "",
                "accuracy_drop_mean": safe_mean(acc_drop) if success_rows else "",
                "accuracy_drop_std": safe_std(acc_drop) if success_rows else "",
                "macro_f1_drop_mean": safe_mean(f1_drop) if success_rows else "",
                "macro_f1_drop_std": safe_std(f1_drop) if success_rows else "",
                "failed_seeds": ";".join([r["seed"] for r in failed_rows]),
            }
        )

    deletion_order = {"original": 0, "high_score": 1, "random": 2, "low_score": 3}
    score_order = {"none": 0, "evidence_norm": 1}

    summary_rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            score_order.get(str(r["score_basis"]), 99),
            float(r["deletion_ratio"]),
            deletion_order.get(str(r["deletion_type"]), 99),
        )
    )

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_average_deletion_curve(
    summary_csv: str | Path,
    output_pdf: str | Path,
    output_png: str | Path,
) -> None:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping plot generation because a plotting dependency is missing: {exc}")
        return

    df = pd.read_csv(summary_csv)
    df = df[
        (df["score_basis"] == "evidence_norm")
        & (df["deletion_type"].isin(["high_score", "random", "low_score"]))
    ].copy()

    if df.empty:
        print("Skipping plot generation because no evidence_norm deletion rows were found.")
        return

    for col in ["deletion_ratio", "macro_f1_drop_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    curve = (
        df.groupby(["deletion_type", "deletion_ratio"], as_index=False)["macro_f1_drop_mean"]
        .mean()
        .sort_values(["deletion_type", "deletion_ratio"])
    )

    label_map = {
        "high_score": "High-evidence deletion",
        "random": "Random deletion",
        "low_score": "Low-evidence deletion",
    }

    plt.figure(figsize=(7.2, 4.6))
    for deletion_type in ["high_score", "random", "low_score"]:
        sub = curve[curve["deletion_type"] == deletion_type].sort_values("deletion_ratio")
        if sub.empty:
            continue
        plt.plot(
            sub["deletion_ratio"] * 100.0,
            sub["macro_f1_drop_mean"],
            marker="o",
            label=label_map[deletion_type],
        )

    plt.xlabel("Deletion ratio (%)")
    plt.ylabel("Macro-F1 drop")
    plt.title("Evidence-Level Deletion Across Deletion Ratios")
    plt.legend()
    plt.tight_layout()

    output_pdf = Path(output_pdf)
    output_png = Path(output_png)
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Average deletion-curve figure saved to:", output_pdf)
    print("Average deletion-curve figure saved to:", output_png)


def build_config_from_args(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.results_dir = Path(args.results_dir)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        cfg.device = torch.device(args.device)
    else:
        cfg.device = torch.device(DEFAULT_DEVICE)

    if args.datasets is not None:
        cfg.dataset_names = args.datasets
    if args.seeds is not None:
        cfg.seeds = args.seeds

    cfg.max_epochs = args.max_epochs
    cfg.patience = args.patience
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.hidden_dim = args.hidden_dim
    cfg.latent_dim = args.latent_dim
    cfg.random_deletion_repeats = args.random_deletion_repeats

    (
        cfg.detail_csv,
        cfg.summary_csv,
        cfg.average_curve_pdf,
        cfg.average_curve_png,
    ) = make_results_file_names(cfg.results_dir)

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)

    print("RUNNING EVIDENCE-LEVEL DELETION CURVE EXPERIMENT")
    print("Datasets:", cfg.dataset_names)
    print("Seeds:", cfg.seeds)
    print("Model:", cfg.model_name)
    print("Score bases:", cfg.score_bases)
    print("Deletion ratios:", cfg.deletion_ratios)
    print("Random deletion repeats:", cfg.random_deletion_repeats)
    print("Validation split:", cfg.use_validation_split, "| val_size:", cfg.val_size)
    print("Device:", cfg.device)
    print("Detail CSV:", cfg.detail_csv)
    print("Summary CSV:", cfg.summary_csv)
    print("Average curve PDF:", cfg.average_curve_pdf)
    print("Average curve PNG:", cfg.average_curve_png)

    for dataset_name in cfg.dataset_names:
        for seed in cfg.seeds:
            run_one_dataset_seed(cfg, dataset_name, seed)

    summarize_results(cfg.detail_csv, cfg.summary_csv)
    plot_average_deletion_curve(
        summary_csv=cfg.summary_csv,
        output_pdf=cfg.average_curve_pdf,
        output_png=cfg.average_curve_png,
    )

    print("\nEvidence-level deletion curve experiment finished.")
    print("Detail results saved to:", cfg.detail_csv)
    print("Summary results saved to:", cfg.summary_csv)
    print("Average curve PDF saved to:", cfg.average_curve_pdf)
    print("Average curve PNG saved to:", cfg.average_curve_png)


if __name__ == "__main__":
    main()
