"""Perturbed-input evaluation for the evidence-gated SSM paper.

This script trains PlainSSM and Proposed_Unnormalized_Base on clean UCR training
splits and evaluates them on clean and perturbed UCR test inputs. The perturbation
conditions are Gaussian noise, random masking, and local interval corruption.

The default dataset list reproduces the five-dataset perturbation experiment used
in Section 5.8 of the manuscript. Use --datasets after adding a command-line
interface, or edit TrainConfig.dataset_names, to run additional datasets.
"""

from __future__ import annotations

from datetime import datetime
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from process_datasets import (
    load_dataset,
    get_integer_labels_from_onehot,
    device,
)

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("Results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "perturbation_type",
    "perturbation_level",
    "repeat",
    "status",
    "error_message",
    "accuracy",
    "macro_f1",
    "accuracy_drop",
    "macro_f1_drop",
    "clean_accuracy",
    "clean_macro_f1",
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
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "perturbation_type",
    "perturbation_level",
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


@dataclass
class TrainConfig:
    dataset_names: List[str] = field(default_factory=lambda: [
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
    ])

    seeds: List[int] = field(default_factory=lambda: [2025, 2026, 2027, 2028, 2029])

    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 50
    patience: int = 10
    hidden_dim: int = 64
    latent_dim: int = 64
    clip_grad_max_norm: float = 0.5

    use_validation_split: bool = True
    val_size: float = 0.2

    # Perturbation levels.
    # Gaussian noise is additive noise after per-series normalization.
    gaussian_sigmas: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])

    # Random masking sets randomly selected time steps to 0.
    random_mask_ratios: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])

    # Local interval corruption selects one contiguous interval and sets it to 0.
    interval_ratios: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])

    # Number of stochastic repeats for each perturbation condition.
    perturbation_repeats: int = 3

    detail_csv: str = ""
    summary_csv: str = ""


MODEL_CONFIGS = [
    {
        "model_name": "PlainSSM",
        "model_type": "plain",
    },
    {
        "model_name": "Proposed_Unnormalized_Base",
        "model_type": "gated",
    },
]


# ============================================================
# Utilities
# ============================================================

def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_results_csv_names() -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_csv = RESULTS_DIR / f"perturbed_input_evaluation_detail_{timestamp}.csv"
    summary_csv = RESULTS_DIR / f"perturbed_input_evaluation_summary_{timestamp}.csv"
    return str(detail_csv), str(summary_csv)


def append_row(csv_path: str | Path, row: Dict, columns: List[str]) -> None:
    csv_path = Path(csv_path)
    exists = csv_path.exists()
    full_row = {col: row.get(col, "") for col in columns}

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(full_row)


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

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

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


# ============================================================
# Models
# ============================================================

class PlainSSMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor):
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

        return logits, {"z": z}


class ProposedUnnormalizedSSM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x: torch.Tensor):
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

        u = (alpha.unsqueeze(-1) * z).sum(dim=1)
        logits = self.classifier(u)

        return logits, {
            "z": z,
            "alpha": alpha,
            "A": self.A,
        }


def build_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    n_classes: int,
) -> nn.Module:
    if model_type == "plain":
        return PlainSSMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_classes=n_classes,
        ).to(device)

    if model_type == "gated":
        return ProposedUnnormalizedSSM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_classes=n_classes,
        ).to(device)

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# Training and evaluation
# ============================================================

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
    model_name: str,
    seed: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_metric = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    epochs_run = 0
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
                raise ValueError(
                    f"Non-finite loss detected for {model_name} on {dataset_name}, seed {seed}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.clip_grad_max_norm,
            )
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader)
        val_metric = val_metrics["macro_f1"]
        epochs_run = epoch + 1

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
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

    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "train_time_sec": train_time,
    }


def get_gate_statistics(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    alpha_values: List[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, _ in loader:
            _, aux = model(X_batch)
            if "alpha" not in aux:
                return {
                    "alpha_mean": "",
                    "alpha_std": "",
                    "alpha_min": "",
                    "alpha_max": "",
                }
            alpha_values.append(aux["alpha"].detach().cpu().reshape(-1))

    if not alpha_values:
        return {
            "alpha_mean": "",
            "alpha_std": "",
            "alpha_min": "",
            "alpha_max": "",
        }

    alpha = torch.cat(alpha_values)
    return {
        "alpha_mean": float(alpha.mean()),
        "alpha_std": float(alpha.std()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
    }


# ============================================================
# Perturbations
# ============================================================

def apply_gaussian_noise(
    X: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(
        X.shape,
        device=X.device,
        dtype=X.dtype,
        generator=generator,
    ) * sigma
    return X + noise


def apply_random_masking(
    X: torch.Tensor,
    ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    X_pert = X.clone()
    batch_size, seq_len = X.shape
    k = int(round(ratio * seq_len))
    k = max(1, min(k, seq_len))

    for i in range(batch_size):
        idx = torch.randperm(seq_len, device=X.device, generator=generator)[:k]
        X_pert[i, idx] = 0.0

    return X_pert


def apply_local_interval_corruption(
    X: torch.Tensor,
    ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    X_pert = X.clone()
    batch_size, seq_len = X.shape
    interval_len = int(round(ratio * seq_len))
    interval_len = max(1, min(interval_len, seq_len))

    max_start = seq_len - interval_len

    for i in range(batch_size):
        if max_start <= 0:
            start = 0
        else:
            start = int(torch.randint(
                low=0,
                high=max_start + 1,
                size=(1,),
                device=X.device,
                generator=generator,
            ).item())
        X_pert[i, start:start + interval_len] = 0.0

    return X_pert


def perturb_batch(
    X: torch.Tensor,
    perturbation_type: str,
    perturbation_level: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if perturbation_type == "clean":
        return X

    if perturbation_type == "gaussian_noise":
        return apply_gaussian_noise(X, sigma=perturbation_level, generator=generator)

    if perturbation_type == "random_masking":
        return apply_random_masking(X, ratio=perturbation_level, generator=generator)

    if perturbation_type == "local_interval_corruption":
        return apply_local_interval_corruption(X, ratio=perturbation_level, generator=generator)

    raise ValueError(f"Unknown perturbation_type: {perturbation_type}")


def evaluate_perturbed_loader(
    model: nn.Module,
    loader: DataLoader,
    perturbation_type: str,
    perturbation_level: float,
    seed: int,
    repeat: int,
) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 10000 * repeat + 123)

    start_time = time.time()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_pert = perturb_batch(
                X=X_batch,
                perturbation_type=perturbation_type,
                perturbation_level=perturbation_level,
                generator=generator,
            )

            logits, _ = model(X_pert)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y_batch.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    inference_time = time.time() - start_time

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "inference_time_sec": inference_time,
    }


def get_perturbation_grid(cfg: TrainConfig) -> List[Tuple[str, float]]:
    grid: List[Tuple[str, float]] = [("clean", 0.0)]

    for sigma in cfg.gaussian_sigmas:
        grid.append(("gaussian_noise", sigma))

    for ratio in cfg.random_mask_ratios:
        grid.append(("random_masking", ratio))

    for ratio in cfg.interval_ratios:
        grid.append(("local_interval_corruption", ratio))

    return grid


# ============================================================
# Experiment
# ============================================================

def write_result_row(
    cfg: TrainConfig,
    dataset_name: str,
    model_name: str,
    seed: int,
    perturbation_type: str,
    perturbation_level: float,
    repeat: int,
    eval_result: Dict[str, float],
    clean_accuracy: float,
    clean_macro_f1: float,
    train_info: Dict[str, float],
    n_params: int,
    gate_stats: Dict[str, float],
) -> None:
    row = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level,
        "repeat": repeat,
        "status": "success",
        "error_message": "",
        "accuracy": eval_result["accuracy"],
        "macro_f1": eval_result["macro_f1"],
        "accuracy_drop": clean_accuracy - eval_result["accuracy"],
        "macro_f1_drop": clean_macro_f1 - eval_result["macro_f1"],
        "clean_accuracy": clean_accuracy,
        "clean_macro_f1": clean_macro_f1,
        "best_epoch": train_info["best_epoch"],
        "epochs_run": train_info["epochs_run"],
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
    row.update(gate_stats)

    append_row(cfg.detail_csv, row, DETAIL_COLUMNS)


def write_error_rows(
    cfg: TrainConfig,
    dataset_name: str,
    model_name: str,
    seed: int,
    error_message: str,
) -> None:
    grid = get_perturbation_grid(cfg)

    for perturbation_type, perturbation_level in grid:
        repeats = 1 if perturbation_type == "clean" else cfg.perturbation_repeats

        for repeat in range(repeats):
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "perturbation_type": perturbation_type,
                "perturbation_level": perturbation_level,
                "repeat": repeat,
                "status": "failed",
                "error_message": error_message,
                "accuracy": "",
                "macro_f1": "",
                "accuracy_drop": "",
                "macro_f1_drop": "",
                "clean_accuracy": "",
                "clean_macro_f1": "",
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
                "alpha_mean": "",
                "alpha_std": "",
                "alpha_min": "",
                "alpha_max": "",
            }
            append_row(cfg.detail_csv, row, DETAIL_COLUMNS)


def run_one_dataset_seed_model(
    cfg: TrainConfig,
    dataset_name: str,
    seed: int,
    model_cfg: Dict[str, str],
) -> None:
    model_name = model_cfg["model_name"]
    model_type = model_cfg["model_type"]

    print("\n" + "=" * 90)
    print(f"Perturbed-input evaluation | dataset={dataset_name} | model={model_name} | seed={seed}")
    print("=" * 90)

    set_global_seed(seed)

    try:
        X_train, y_train_oh, X_test, y_test_oh = load_dataset(dataset_name)
        y_train = get_integer_labels_from_onehot(y_train_oh)
        y_test = get_integer_labels_from_onehot(y_test_oh)

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

        model = build_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            n_classes=n_classes,
        )

        model, train_info = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
        )

        n_params = count_parameters(model)
        gate_stats = get_gate_statistics(model, test_loader)

        clean_eval = evaluate_perturbed_loader(
            model=model,
            loader=test_loader,
            perturbation_type="clean",
            perturbation_level=0.0,
            seed=seed,
            repeat=0,
        )

        clean_accuracy = clean_eval["accuracy"]
        clean_macro_f1 = clean_eval["macro_f1"]

        write_result_row(
            cfg=cfg,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            perturbation_type="clean",
            perturbation_level=0.0,
            repeat=0,
            eval_result=clean_eval,
            clean_accuracy=clean_accuracy,
            clean_macro_f1=clean_macro_f1,
            train_info=train_info,
            n_params=n_params,
            gate_stats=gate_stats,
        )

        print(
            f"[{dataset_name}][{model_name}][seed={seed}] clean | "
            f"Acc={clean_accuracy:.4f} | MacroF1={clean_macro_f1:.4f}"
        )

        for perturbation_type, perturbation_level in get_perturbation_grid(cfg):
            if perturbation_type == "clean":
                continue

            for repeat in range(cfg.perturbation_repeats):
                eval_result = evaluate_perturbed_loader(
                    model=model,
                    loader=test_loader,
                    perturbation_type=perturbation_type,
                    perturbation_level=perturbation_level,
                    seed=seed,
                    repeat=repeat,
                )

                write_result_row(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    perturbation_type=perturbation_type,
                    perturbation_level=perturbation_level,
                    repeat=repeat,
                    eval_result=eval_result,
                    clean_accuracy=clean_accuracy,
                    clean_macro_f1=clean_macro_f1,
                    train_info=train_info,
                    n_params=n_params,
                    gate_stats=gate_stats,
                )

                print(
                    f"[{dataset_name}][{model_name}][seed={seed}] "
                    f"{perturbation_type}={perturbation_level} repeat={repeat} | "
                    f"Acc={eval_result['accuracy']:.4f} "
                    f"(drop={clean_accuracy - eval_result['accuracy']:.4f}) | "
                    f"MacroF1={eval_result['macro_f1']:.4f} "
                    f"(drop={clean_macro_f1 - eval_result['macro_f1']:.4f})"
                )

    except Exception as e:
        print(f"[{dataset_name}][{model_name}][seed={seed}] FAILED: {e}")
        write_error_rows(
            cfg=cfg,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            error_message=str(e),
        )


# ============================================================
# Summary
# ============================================================

def summarize_results(detail_csv: str | Path, summary_csv: str | Path) -> None:
    detail_csv = Path(detail_csv)
    rows: List[Dict[str, str]] = []

    with open(detail_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}

    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["perturbation_type"],
            row["perturbation_level"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []

    for (dataset, model, perturbation_type, perturbation_level), group_rows in groups.items():
        success_rows = [r for r in group_rows if r["status"] == "success"]
        failed_rows = [r for r in group_rows if r["status"] != "success"]

        acc = np.array([to_float(r["accuracy"]) for r in success_rows], dtype=float)
        f1 = np.array([to_float(r["macro_f1"]) for r in success_rows], dtype=float)
        acc_drop = np.array([to_float(r["accuracy_drop"]) for r in success_rows], dtype=float)
        f1_drop = np.array([to_float(r["macro_f1_drop"]) for r in success_rows], dtype=float)

        failed_seeds = ";".join(sorted(set([r["seed"] for r in failed_rows])))

        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "perturbation_type": perturbation_type,
            "perturbation_level": perturbation_level,
            "n_success": len(success_rows),
            "n_failed": len(failed_rows),
            "accuracy_mean": safe_mean(acc) if len(success_rows) > 0 else "",
            "accuracy_std": safe_std(acc) if len(success_rows) > 0 else "",
            "macro_f1_mean": safe_mean(f1) if len(success_rows) > 0 else "",
            "macro_f1_std": safe_std(f1) if len(success_rows) > 0 else "",
            "accuracy_drop_mean": safe_mean(acc_drop) if len(success_rows) > 0 else "",
            "accuracy_drop_std": safe_std(acc_drop) if len(success_rows) > 0 else "",
            "macro_f1_drop_mean": safe_mean(f1_drop) if len(success_rows) > 0 else "",
            "macro_f1_drop_std": safe_std(f1_drop) if len(success_rows) > 0 else "",
            "failed_seeds": failed_seeds,
        })

    perturb_order = {
        "clean": 0,
        "gaussian_noise": 1,
        "random_masking": 2,
        "local_interval_corruption": 3,
    }

    summary_rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            str(r["model"]),
            perturb_order.get(str(r["perturbation_type"]), 99),
            float(r["perturbation_level"]),
        )
    )

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = TrainConfig()
    cfg.detail_csv, cfg.summary_csv = make_results_csv_names()

    print("RUNNING PERTURBED-INPUT EVALUATION")
    print("Datasets:", cfg.dataset_names)
    print("Seeds:", cfg.seeds)
    print("Models:", [m["model_name"] for m in MODEL_CONFIGS])
    print("Gaussian sigmas:", cfg.gaussian_sigmas)
    print("Random mask ratios:", cfg.random_mask_ratios)
    print("Interval corruption ratios:", cfg.interval_ratios)
    print("Perturbation repeats:", cfg.perturbation_repeats)
    print("Validation split:", cfg.use_validation_split, "| val_size:", cfg.val_size)
    print("Device:", device)
    print("Detail CSV:", cfg.detail_csv)
    print("Summary CSV:", cfg.summary_csv)

    for dataset_name in cfg.dataset_names:
        for seed in cfg.seeds:
            for model_cfg in MODEL_CONFIGS:
                run_one_dataset_seed_model(cfg, dataset_name, seed, model_cfg)

    summarize_results(cfg.detail_csv, cfg.summary_csv)

    print("\nPerturbed-input evaluation finished.")
    print("Detail results saved to:", cfg.detail_csv)
    print("Summary results saved to:", cfg.summary_csv)


if __name__ == "__main__":
    main()
