from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from process_datasets import load_dataset, get_integer_labels_from_onehot


# =============================================================================
# Qualitative evidence-profile experiment
# =============================================================================

DEFAULT_DATASETS: List[str] = [
    "ECG5000",
    "ElectricDevices",
]

DEFAULT_SEED = 2025


@dataclass
class TrainConfig:
    """Configuration for qualitative evidence-profile extraction."""

    dataset_names: List[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    seed: int = DEFAULT_SEED

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

    top_fraction: float = 0.10
    delete_fraction: float = 0.10
    n_examples_per_dataset: int = 1
    selection_rule: str = "largest_confidence_drop"

    results_dir: Path = Path("Results")
    output_dir: Path = Path("")
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
    """Set NumPy and PyTorch random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(results_dir: Path) -> Path:
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / f"qualitative_evidence_profiles_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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
    """Create train, validation, and test DataLoaders."""
    X_train = X_train.detach().cpu()
    y_train = y_train.detach().cpu().long()
    X_test = X_test.detach().cpu()
    y_test = y_test.detach().cpu().long()

    if use_validation_split:
        indices = np.arange(len(X_train))
        labels = y_train.numpy()

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

        train_ds = TensorDataset(X_train[train_idx], y_train[train_idx])
        val_ds = TensorDataset(X_train[val_idx], y_train[val_idx])
    else:
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_test, y_test)

    test_ds = TensorDataset(X_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Model
# =============================================================================

class ProposedUnnormalizedSSM(nn.Module):
    """Proposed unnormalized evidence-gated state-space classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)

        self.feature_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return latent features z and gate values alpha."""
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
        """Classify from unnormalized additive evidence."""
        u = (alpha.unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def classify_from_masked_evidence(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Classify after deleting selected evidence terms."""
        u = ((alpha * keep_mask).unsqueeze(-1) * z).sum(dim=1)
        return self.classifier(u)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, alpha = self.encode(x)
        logits = self.classify_from_evidence(z, alpha)
        return logits, {"z": z, "alpha": alpha, "A": self.A}


# =============================================================================
# Training and evaluation
# =============================================================================

def evaluate_loader(model: nn.Module, loader: DataLoader, run_device: torch.device) -> Dict[str, float]:
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    dataset_name: str,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train one proposed model with early stopping on validation macro-F1."""
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
            X_batch = X_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = loss_fn(logits, y_batch)

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected on {dataset_name}, seed={cfg.seed}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_max_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader, cfg.device)
        val_metric = val_metrics["macro_f1"]

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{dataset_name}][{cfg.model_name}][seed={cfg.seed}] "
            f"Epoch {epoch + 1:03d} | "
            f"TrainLoss={np.mean(epoch_losses):.4f} | "
            f"ValAcc={val_metrics['accuracy']:.4f} | "
            f"ValMacroF1={val_metrics['macro_f1']:.4f}"
        )

        if patience_counter >= cfg.patience:
            print(f"[{dataset_name}][{cfg.model_name}][seed={cfg.seed}] Early stopping triggered.")
            break

    train_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_epoch": best_epoch, "train_time_sec": train_time}


# =============================================================================
# Evidence profile extraction
# =============================================================================

def get_top_indices(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return top-k temporal indices according to a score vector."""
    seq_len = scores.shape[0]
    k = int(round(fraction * seq_len))
    k = max(1, min(k, seq_len))
    return torch.topk(scores, k=k, largest=True).indices


def make_keep_mask_for_single(seq_len: int, delete_idx: torch.Tensor, run_device: torch.device) -> torch.Tensor:
    """Create a single-example keep mask where selected indices are deleted."""
    keep_mask = torch.ones(1, seq_len, device=run_device)
    keep_mask[0, delete_idx] = 0.0
    return keep_mask


def collect_candidate_profiles(
    model: ProposedUnnormalizedSSM,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    dataset_name: str,
    cfg: TrainConfig,
) -> List[Dict]:
    """Collect correctly classified test examples with evidence profiles."""
    model.eval()
    candidates: List[Dict] = []

    X_test = X_test.detach().cpu()
    y_test = y_test.detach().cpu().long()

    with torch.no_grad():
        for idx in range(len(X_test)):
            x = X_test[idx:idx + 1].to(cfg.device)
            y = int(y_test[idx].item())

            z, alpha = model.encode(x)
            logits = model.classify_from_evidence(z, alpha)
            prob = torch.softmax(logits, dim=1)

            pred = int(torch.argmax(prob, dim=1).item())
            clean_conf = float(prob[0, pred].detach().cpu())

            if pred != y:
                continue

            evidence = alpha.unsqueeze(-1) * z
            evidence_norm = torch.linalg.vector_norm(evidence, ord=2, dim=-1)[0]

            top_idx = get_top_indices(evidence_norm, cfg.delete_fraction)
            keep_mask = make_keep_mask_for_single(
                seq_len=evidence_norm.shape[0],
                delete_idx=top_idx,
                run_device=cfg.device,
            )

            deleted_logits = model.classify_from_masked_evidence(z, alpha, keep_mask)
            deleted_prob = torch.softmax(deleted_logits, dim=1)

            deleted_pred = int(torch.argmax(deleted_prob, dim=1).item())
            deleted_conf_original_pred = float(deleted_prob[0, pred].detach().cpu())
            deleted_conf_new_pred = float(deleted_prob[0, deleted_pred].detach().cpu())

            candidates.append({
                "dataset": dataset_name,
                "index": idx,
                "x": x[0].detach().cpu().numpy(),
                "true_label": y,
                "pred_label": pred,
                "clean_conf": clean_conf,
                "deleted_pred_label": deleted_pred,
                "deleted_conf_original_pred": deleted_conf_original_pred,
                "deleted_conf_new_pred": deleted_conf_new_pred,
                "confidence_drop": clean_conf - deleted_conf_original_pred,
                "alpha": alpha[0].detach().cpu().numpy(),
                "evidence_norm": evidence_norm.detach().cpu().numpy(),
                "top_idx": top_idx.detach().cpu().numpy(),
            })

    return candidates


def select_profiles(candidates: List[Dict], cfg: TrainConfig) -> List[Dict]:
    """Select representative qualitative profiles."""
    if not candidates:
        return []

    if cfg.selection_rule == "largest_confidence_drop":
        sorted_candidates = sorted(candidates, key=lambda item: item["confidence_drop"], reverse=True)
    elif cfg.selection_rule == "highest_confidence":
        sorted_candidates = sorted(candidates, key=lambda item: item["clean_conf"], reverse=True)
    else:
        raise ValueError(f"Unknown selection_rule: {cfg.selection_rule}")

    return sorted_candidates[:cfg.n_examples_per_dataset]


# =============================================================================
# Plotting
# =============================================================================

def normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Normalize a vector to [0, 1] for visualization."""
    values = np.asarray(values, dtype=float)
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)

    if not np.isfinite(v_min) or not np.isfinite(v_max) or abs(v_max - v_min) < 1e-12:
        return np.zeros_like(values)

    return (values - v_min) / (v_max - v_min)


def contiguous_regions(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Convert sorted temporal indices to contiguous [start, end] regions."""
    if len(indices) == 0:
        return []

    indices = np.sort(indices)
    regions: List[Tuple[int, int]] = []

    start = int(indices[0])
    prev = int(indices[0])

    for idx_raw in indices[1:]:
        idx = int(idx_raw)
        if idx == prev + 1:
            prev = idx
        else:
            regions.append((start, prev))
            start = idx
            prev = idx

    regions.append((start, prev))
    return regions


def plot_profile(profile: Dict, output_dir: Path, cfg: TrainConfig) -> Tuple[str, str]:
    """Save one qualitative evidence-profile figure as PDF and PNG."""
    x = profile["x"]
    evidence_norm = normalize_to_unit_interval(profile["evidence_norm"])
    alpha = normalize_to_unit_interval(profile["alpha"])
    top_idx = profile["top_idx"]
    time_axis = np.arange(len(x))

    regions = contiguous_regions(top_idx)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )

    axes[0].plot(time_axis, x)
    axes[0].set_ylabel("Input")
    axes[0].set_title(
        f"{profile['dataset']} | test index {profile['index']} | "
        f"true={profile['true_label']}, pred={profile['pred_label']}"
    )

    axes[1].plot(time_axis, evidence_norm)
    axes[1].scatter(top_idx, evidence_norm[top_idx], s=18)
    axes[1].set_ylabel("Evidence norm")
    axes[1].set_title("Evidence score profile")

    axes[2].plot(time_axis, alpha)
    axes[2].set_ylabel("Gate")
    axes[2].set_xlabel("Time index")
    axes[2].set_title("Gate activation profile")

    for ax in axes:
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.18)

    fig.text(
        0.01,
        0.01,
        (
            f"Clean confidence={profile['clean_conf']:.3f}; "
            f"confidence after deleting top {cfg.delete_fraction:.0%} evidence="
            f"{profile['deleted_conf_original_pred']:.3f}; "
            f"confidence drop={profile['confidence_drop']:.3f}; "
            f"deleted prediction={profile['deleted_pred_label']} "
            f"(confidence={profile['deleted_conf_new_pred']:.3f})"
        ),
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    safe_dataset = str(profile["dataset"]).replace("/", "_")
    base_name = (
        f"qualitative_evidence_profile_{safe_dataset}"
        f"_idx{profile['index']}"
        f"_top{int(cfg.top_fraction * 100)}"
    )

    pdf_path = output_dir / f"{base_name}.pdf"
    png_path = output_dir / f"{base_name}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(pdf_path), str(png_path)


def write_selected_profiles_csv(selected_profiles: List[Dict], output_dir: Path) -> str:
    """Write metadata for selected qualitative examples."""
    csv_path = output_dir / "selected_qualitative_profiles.csv"

    columns = [
        "dataset",
        "index",
        "true_label",
        "pred_label",
        "clean_conf",
        "deleted_pred_label",
        "deleted_conf_original_pred",
        "deleted_conf_new_pred",
        "confidence_drop",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for profile in selected_profiles:
            writer.writerow({col: profile.get(col, "") for col in columns})

    return str(csv_path)


# =============================================================================
# Main experiment
# =============================================================================

def run_one_dataset(cfg: TrainConfig, dataset_name: str) -> List[Dict]:
    """Train the proposed model and export qualitative profiles for one dataset."""
    print("\n" + "=" * 80)
    print(f"Qualitative evidence profiles | dataset={dataset_name} | seed={cfg.seed}")
    print("=" * 80)

    set_global_seed(cfg.seed)

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
        seed=cfg.seed,
        use_validation_split=cfg.use_validation_split,
        val_size=cfg.val_size,
    )

    set_global_seed(cfg.seed)

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
    )

    test_metrics = evaluate_loader(model, test_loader, cfg.device)
    print(
        f"[{dataset_name}] Test Acc={test_metrics['accuracy']:.4f} | "
        f"Test MacroF1={test_metrics['macro_f1']:.4f} | "
        f"best_epoch={train_info['best_epoch']} | "
        f"n_params={count_parameters(model)}"
    )

    candidates = collect_candidate_profiles(
        model=model,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
        cfg=cfg,
    )

    selected = select_profiles(candidates, cfg)

    if not selected:
        print(f"[{dataset_name}] No correctly classified candidates found.")
        return []

    for profile in selected:
        pdf_path, png_path = plot_profile(profile, cfg.output_dir, cfg)
        print(
            f"[{dataset_name}] saved profile idx={profile['index']} | "
            f"confidence_drop={profile['confidence_drop']:.4f} | "
            f"PDF={pdf_path} | PNG={png_path}"
        )

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative evidence profiles for selected UCR datasets."
    )

    parser.add_argument("--results-dir", type=str, default="Results")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--clip-grad-max-norm", type=float, default=0.5)

    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--delete-fraction", type=float, default=0.10)
    parser.add_argument("--n-examples-per-dataset", type=int, default=1)
    parser.add_argument(
        "--selection-rule",
        type=str,
        default="largest_confidence_drop",
        choices=["largest_confidence_drop", "highest_confidence"],
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--no-validation-split", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        dataset_names=list(args.datasets),
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        clip_grad_max_norm=args.clip_grad_max_norm,
        use_validation_split=not args.no_validation_split,
        val_size=args.val_size,
        top_fraction=args.top_fraction,
        delete_fraction=args.delete_fraction,
        n_examples_per_dataset=args.n_examples_per_dataset,
        selection_rule=args.selection_rule,
        results_dir=Path(args.results_dir),
        device=resolve_device(args.device),
    )
    cfg.output_dir = make_output_dir(cfg.results_dir)

    print("RUNNING QUALITATIVE EVIDENCE-PROFILE EXPERIMENT")
    print("Datasets:", cfg.dataset_names)
    print("Seed:", cfg.seed)
    print("Model:", cfg.model_name)
    print("Top fraction highlighted:", cfg.top_fraction)
    print("Delete fraction for confidence drop:", cfg.delete_fraction)
    print("Examples per dataset:", cfg.n_examples_per_dataset)
    print("Selection rule:", cfg.selection_rule)
    print("Validation split:", cfg.use_validation_split, "| val_size:", cfg.val_size)
    print("Device:", cfg.device)
    print("Output directory:", cfg.output_dir)

    all_selected: List[Dict] = []

    for dataset_name in cfg.dataset_names:
        try:
            selected = run_one_dataset(cfg, dataset_name)
            all_selected.extend(selected)
        except Exception as exc:
            print(f"[{dataset_name}] FAILED: {exc}")

    csv_path = write_selected_profiles_csv(all_selected, cfg.output_dir)

    print("\nQualitative evidence-profile experiment finished.")
    print("Output directory:", cfg.output_dir)
    print("Selected profiles CSV:", csv_path)


if __name__ == "__main__":
    main()
