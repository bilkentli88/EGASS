from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Synthetic evidence localization experiment
# =============================================================================

DEFAULT_SCORE_NAMES: List[str] = [
    "random",
    "latent_norm",
    "gate_alpha",
    "evidence_norm",
    "class_logit",
    "margin_logit",
]

DISPLAY_NAMES: Dict[str, str] = {
    "random": "Random ranking",
    "latent_norm": r"Latent norm $\|z_t\|_2$",
    "gate_alpha": r"Gate activation $\alpha_t$",
    "evidence_norm": r"Evidence norm $\|e_t\|_2$",
    "class_logit": r"Class-logit score $w_{\hat c}^\top e_t$",
    "margin_logit": r"Margin-logit score $(w_{\hat c}-w_{c'})^\top e_t$",
}


@dataclass
class SyntheticConfig:
    seed: int = 2025
    device: torch.device = torch.device("cpu")
    output_dir: Path = Path("synthetic_results")

    n_samples: int = 3000
    length: int = 100
    motif_len: int = 15
    noise_std: float = 0.50
    motif_amp: float = 2.0
    add_distractor: bool = True
    distractor_amp: float = 2.5
    distractor_len: int = 10

    hidden_dim: int = 64
    latent_dim: int = 64
    batch_size: int = 64
    lr: float = 5e-4
    max_epochs: int = 80
    patience: int = 10
    grad_clip: float = 0.5
    lambda_sparse: float = 0.0

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    evidence_ratios: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)


# =============================================================================
# Reproducibility and device handling
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    """Resolve requested computation device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return device


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Synthetic data generation
# =============================================================================

def generate_sine_motif(length: int) -> np.ndarray:
    """Generate the discriminative sine motif."""
    t = np.linspace(0, 2 * np.pi, length)
    return np.sin(t).astype(np.float32)


def generate_synthetic_dataset(
    n_samples: int,
    length: int,
    motif_len: int,
    noise_std: float,
    motif_amp: float,
    add_distractor: bool,
    distractor_amp: float,
    distractor_len: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a binary synthetic dataset.

    Class-1 samples contain a localized sine motif at a random location.
    Both classes may include high-amplitude distractor segments that are not
    correlated with the label.
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0.0, noise_std, size=(n_samples, length)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    true_mask = np.zeros((n_samples, length), dtype=np.int64)

    motif = generate_sine_motif(motif_len)

    for i in range(n_samples):
        if y[i] == 1:
            start = int(rng.integers(5, length - motif_len - 5))
            end = start + motif_len

            X[i, start:end] += motif_amp * motif
            true_mask[i, start:end] = 1

        if add_distractor:
            d_start = int(rng.integers(5, length - distractor_len - 5))
            d_end = d_start + distractor_len

            if y[i] == 1:
                attempts = 0
                while true_mask[i, d_start:d_end].sum() > 0 and attempts < 30:
                    d_start = int(rng.integers(5, length - distractor_len - 5))
                    d_end = d_start + distractor_len
                    attempts += 1

            sign = float(rng.choice([-1.0, 1.0]))
            X[i, d_start:d_end] += sign * distractor_amp

    return X[..., None].astype(np.float32), y, true_mask


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    """Split arrays into train, validation, and test partitions."""
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        X[train_idx], y[train_idx], mask[train_idx],
        X[val_idx], y[val_idx], mask[val_idx],
        X[test_idx], y[test_idx], mask[test_idx],
    )


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# =============================================================================
# Model
# =============================================================================

class EvidenceGatedSSM(nn.Module):
    """Simple evidence-gated state-space classifier."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 64, num_classes: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.A = nn.Parameter(0.05 * torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Linear(input_dim, hidden_dim, bias=False)

        self.z_layer = nn.Linear(hidden_dim + input_dim, latent_dim)
        self.gate_layer = nn.Linear(latent_dim, 1)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_evidence: bool = False,
    ):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        z_list = []
        alpha_list = []
        evidence_list = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = h @ self.A.T + self.B(x_t)

            z_t = torch.tanh(self.z_layer(torch.cat([h, x_t], dim=1)))
            alpha_t = torch.sigmoid(self.gate_layer(z_t))
            e_t = alpha_t * z_t

            z_list.append(z_t)
            alpha_list.append(alpha_t.squeeze(-1))
            evidence_list.append(e_t)

        z_seq = torch.stack(z_list, dim=1)
        alpha = torch.stack(alpha_list, dim=1)
        evidence_seq = torch.stack(evidence_list, dim=1)

        u = evidence_seq.sum(dim=1)
        logits = self.classifier(u)

        if return_evidence:
            return logits, alpha, z_seq, evidence_seq

        return logits


# =============================================================================
# Training and evaluation
# =============================================================================

def train_model(
    model: EvidenceGatedSSM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
    grad_clip: float,
    lambda_sparse: float,
) -> EvidenceGatedSSM:
    """Train the synthetic model with early stopping on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        train_losses = []
        train_ce_losses = []
        train_alpha_means = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(run_device)
            y_batch = y_batch.to(run_device)

            optimizer.zero_grad()
            logits, alpha, _, _ = model(X_batch, return_evidence=True)

            ce_loss = F.cross_entropy(logits, y_batch)
            sparse_loss = alpha.mean()
            loss = ce_loss + lambda_sparse * sparse_loss

            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss detected in synthetic experiment.")

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            train_ce_losses.append(float(ce_loss.detach().cpu()))
            train_alpha_means.append(float(sparse_loss.detach().cpu()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(run_device)
                y_batch = y_batch.to(run_device)

                logits, alpha, _, _ = model(X_batch, return_evidence=True)
                ce_loss = F.cross_entropy(logits, y_batch)
                sparse_loss = alpha.mean()
                loss = ce_loss + lambda_sparse * sparse_loss

                val_losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss={np.mean(train_losses):.5f} | "
                f"CE={np.mean(train_ce_losses):.5f} | "
                f"MeanAlpha={np.mean(train_alpha_means):.5f} | "
                f"ValLoss={val_loss:.5f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_classification(
    model: EvidenceGatedSSM,
    test_loader: DataLoader,
    run_device: torch.device,
) -> Tuple[float, float]:
    """Evaluate test accuracy and macro-F1."""
    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(run_device)

            logits = model(X_batch)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

            all_y.append(y_batch.numpy())
            all_pred.append(pred)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


# =============================================================================
# Evidence scores
# =============================================================================

def classifier_aware_scores_for_sample(
    evidence_i: np.ndarray,
    classifier_weights: np.ndarray,
    logits_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute class-logit and margin-logit evidence scores for one sample."""
    pred_class = int(np.argmax(logits_i))

    logits_copy = logits_i.copy()
    logits_copy[pred_class] = -np.inf
    runner_up = int(np.argmax(logits_copy))

    w_pred = classifier_weights[pred_class]
    w_runner_up = classifier_weights[runner_up]

    class_logit_score = evidence_i @ w_pred
    margin_logit_score = evidence_i @ (w_pred - w_runner_up)

    return class_logit_score, margin_logit_score


def get_score_vector(
    score_name: str,
    alpha_i: np.ndarray,
    z_i: np.ndarray,
    evidence_i: np.ndarray,
    classifier_weights: Optional[np.ndarray] = None,
    logits_i: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return one temporal score vector for one sample."""
    if score_name == "random":
        if rng is None:
            rng = np.random.default_rng(2025)
        return rng.random(len(alpha_i))

    if score_name == "latent_norm":
        return np.linalg.norm(z_i, axis=1)

    if score_name == "gate_alpha":
        return alpha_i

    if score_name == "evidence_norm":
        return np.linalg.norm(evidence_i, axis=1)

    if score_name == "class_logit":
        if classifier_weights is None or logits_i is None:
            raise ValueError("classifier_weights and logits_i are required for class_logit.")
        class_score, _ = classifier_aware_scores_for_sample(evidence_i, classifier_weights, logits_i)
        return class_score

    if score_name == "margin_logit":
        if classifier_weights is None or logits_i is None:
            raise ValueError("classifier_weights and logits_i are required for margin_logit.")
        _, margin_score = classifier_aware_scores_for_sample(evidence_i, classifier_weights, logits_i)
        return margin_score

    raise ValueError(f"Unknown score_name: {score_name}")


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k score values."""
    return np.argsort(scores)[-k:]


def localization_metrics_for_one_sample(
    scores: np.ndarray,
    true_mask: np.ndarray,
    k: int,
) -> Optional[Tuple[float, float, float]]:
    """Compute Precision@k, Recall@k, and IoU for one positive sample."""
    top_idx = set(top_k_indices(scores, k))
    true_idx = set(np.where(true_mask == 1)[0])

    if len(true_idx) == 0:
        return None

    intersection = len(top_idx.intersection(true_idx))
    union = len(top_idx.union(true_idx))

    precision = intersection / max(k, 1)
    recall = intersection / max(len(true_idx), 1)
    iou = intersection / max(union, 1)

    return precision, recall, iou


def collect_evidence_arrays(
    model: EvidenceGatedSSM,
    X_test: np.ndarray,
    run_device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one forward pass over the test set and return logits/evidence arrays."""
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(run_device)

    with torch.no_grad():
        logits, alpha, z_seq, evidence_seq = model(X_tensor, return_evidence=True)

    return (
        logits.detach().cpu().numpy(),
        alpha.detach().cpu().numpy(),
        z_seq.detach().cpu().numpy(),
        evidence_seq.detach().cpu().numpy(),
    )


def evaluate_localization(
    model: EvidenceGatedSSM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask_test: np.ndarray,
    run_device: torch.device,
    score_names: Iterable[str],
) -> Tuple[pd.DataFrame, int]:
    """Evaluate top-k localization against the known motif region."""
    logits_np, alpha_np, z_np, evidence_np = collect_evidence_arrays(model, X_test, run_device)
    classifier_weights = model.classifier.weight.detach().cpu().numpy()

    methods = {name: [] for name in score_names}
    rng = np.random.default_rng(2025)

    n_positive = 0

    for i in range(len(X_test)):
        if y_test[i] != 1:
            continue

        true_mask = mask_test[i]
        motif_len = int(true_mask.sum())

        if motif_len == 0:
            continue

        n_positive += 1
        k = motif_len

        for score_name in score_names:
            scores = get_score_vector(
                score_name=score_name,
                alpha_i=alpha_np[i],
                z_i=z_np[i],
                evidence_i=evidence_np[i],
                classifier_weights=classifier_weights,
                logits_i=logits_np[i],
                rng=rng,
            )

            result = localization_metrics_for_one_sample(scores, true_mask, k)
            if result is not None:
                methods[score_name].append(result)

    rows = []
    for score_name, values in methods.items():
        values_array = np.asarray(values, dtype=float)
        rows.append({
            "score": score_name,
            "precision_at_k": float(values_array[:, 0].mean()),
            "recall_at_k": float(values_array[:, 1].mean()),
            "iou": float(values_array[:, 2].mean()),
            "display_name": DISPLAY_NAMES.get(score_name, score_name),
        })

    return pd.DataFrame(rows), n_positive


def evaluate_deletion_insertion(
    model: EvidenceGatedSSM,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_device: torch.device,
    score_names: Iterable[str],
    ratios: Iterable[float],
) -> pd.DataFrame:
    """Evaluate representation-level deletion and insertion curves."""
    logits_np, alpha_np, z_np, evidence_np = collect_evidence_arrays(model, X_test, run_device)

    classifier_weights = model.classifier.weight.detach().cpu().numpy()
    classifier_bias = model.classifier.bias.detach().cpu().numpy()

    y_np = np.asarray(y_test)
    n_samples, seq_len, _ = evidence_np.shape
    rng = np.random.default_rng(2025)

    rows = []

    for score_name in score_names:
        for ratio in ratios:
            k = max(1, int(round(ratio * seq_len)))

            deletion_logits = []
            insertion_logits = []

            for i in range(n_samples):
                scores = get_score_vector(
                    score_name=score_name,
                    alpha_i=alpha_np[i],
                    z_i=z_np[i],
                    evidence_i=evidence_np[i],
                    classifier_weights=classifier_weights,
                    logits_i=logits_np[i],
                    rng=rng,
                )

                top_idx = top_k_indices(scores, k)

                u_full = evidence_np[i].sum(axis=0)
                u_deletion = u_full - evidence_np[i, top_idx, :].sum(axis=0)
                u_insertion = evidence_np[i, top_idx, :].sum(axis=0)

                deletion_logits.append(classifier_weights @ u_deletion + classifier_bias)
                insertion_logits.append(classifier_weights @ u_insertion + classifier_bias)

            deletion_pred = np.stack(deletion_logits).argmax(axis=1)
            insertion_pred = np.stack(insertion_logits).argmax(axis=1)

            rows.append({
                "score": score_name,
                "ratio": ratio,
                "deletion_accuracy": accuracy_score(y_np, deletion_pred),
                "deletion_macro_f1": f1_score(y_np, deletion_pred, average="macro", zero_division=0),
                "insertion_accuracy": accuracy_score(y_np, insertion_pred),
                "insertion_macro_f1": f1_score(y_np, insertion_pred, average="macro", zero_division=0),
                "display_name": DISPLAY_NAMES.get(score_name, score_name),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Export and plotting
# =============================================================================

def make_output_dir(cfg: SyntheticConfig) -> Path:
    """Create an output directory name from key experiment parameters."""
    distractor_tag = "distractor" if cfg.add_distractor else "no_distractor"
    sparse_tag = f"lambda_{cfg.lambda_sparse:g}".replace(".", "p").replace("-", "m")
    out_dir = cfg.output_dir / f"synthetic_{distractor_tag}_{sparse_tag}_seed_{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_experiment_config(
    cfg: SyntheticConfig,
    out_dir: Path,
    accuracy: float,
    macro_f1: float,
    n_positive: int,
) -> None:
    """Save a plain-text experiment configuration and outcome summary."""
    path = out_dir / "experiment_config.txt"

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Synthetic Evidence Localization Experiment\n")
        handle.write("=========================================\n")
        handle.write(f"seed: {cfg.seed}\n")
        handle.write(f"n_samples: {cfg.n_samples}\n")
        handle.write(f"length: {cfg.length}\n")
        handle.write(f"motif_len: {cfg.motif_len}\n")
        handle.write(f"noise_std: {cfg.noise_std}\n")
        handle.write(f"motif_amp: {cfg.motif_amp}\n")
        handle.write(f"add_distractor: {cfg.add_distractor}\n")
        handle.write(f"distractor_amp: {cfg.distractor_amp}\n")
        handle.write(f"distractor_len: {cfg.distractor_len}\n")
        handle.write(f"lambda_sparse: {cfg.lambda_sparse}\n")
        handle.write(f"hidden_dim: {cfg.hidden_dim}\n")
        handle.write(f"latent_dim: {cfg.latent_dim}\n")
        handle.write(f"batch_size: {cfg.batch_size}\n")
        handle.write(f"lr: {cfg.lr}\n")
        handle.write(f"max_epochs: {cfg.max_epochs}\n")
        handle.write(f"patience: {cfg.patience}\n")
        handle.write(f"device: {cfg.device}\n")
        handle.write("\nFinal classification results\n")
        handle.write("----------------------------\n")
        handle.write(f"test_accuracy: {accuracy:.6f}\n")
        handle.write(f"test_macro_f1: {macro_f1:.6f}\n")
        handle.write(f"positive_test_samples_for_localization: {n_positive}\n")


def plot_localization_iou(localization_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot IoU values for the localization experiment."""
    df = localization_df.copy().sort_values("iou", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["display_name"], df["iou"])
    plt.ylabel("IoU")
    plt.xlabel("Evidence score")
    plt.title("Ground-Truth Evidence Localization")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(out_dir / "localization_iou.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "localization_iou.pdf", bbox_inches="tight")
    plt.close()


def plot_curve(curves_df: pd.DataFrame, metric: str, title: str, ylabel: str, out_name: str, out_dir: Path) -> None:
    """Plot a deletion or insertion curve."""
    plt.figure(figsize=(8, 5))

    for score in curves_df["score"].unique():
        sub = curves_df[curves_df["score"] == score].sort_values("ratio")
        display = DISPLAY_NAMES.get(score, score)
        plt.plot(sub["ratio"] * 100, sub[metric], marker="o", label=display)

    plt.xlabel("Selected evidence ratio (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(out_dir / f"{out_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
    plt.close()


def export_results(
    localization_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    cfg: SyntheticConfig,
    out_dir: Path,
    accuracy: float,
    macro_f1: float,
    n_positive: int,
) -> None:
    """Save CSVs, config, and figures."""
    localization_df.to_csv(out_dir / "localization_results.csv", index=False)
    curves_df.to_csv(out_dir / "deletion_insertion_curves.csv", index=False)

    save_experiment_config(cfg, out_dir, accuracy, macro_f1, n_positive)

    plot_localization_iou(localization_df, out_dir)

    plot_curve(
        curves_df=curves_df,
        metric="deletion_accuracy",
        title="Evidence Deletion Curves",
        ylabel="Accuracy after deletion",
        out_name="deletion_accuracy_curves",
        out_dir=out_dir,
    )

    plot_curve(
        curves_df=curves_df,
        metric="deletion_macro_f1",
        title="Evidence Deletion Curves",
        ylabel="Macro-F1 after deletion",
        out_name="deletion_macro_f1_curves",
        out_dir=out_dir,
    )

    plot_curve(
        curves_df=curves_df,
        metric="insertion_accuracy",
        title="Evidence Insertion Curves",
        ylabel="Accuracy after insertion",
        out_name="insertion_accuracy_curves",
        out_dir=out_dir,
    )

    plot_curve(
        curves_df=curves_df,
        metric="insertion_macro_f1",
        title="Evidence Insertion Curves",
        ylabel="Macro-F1 after insertion",
        out_name="insertion_macro_f1_curves",
        out_dir=out_dir,
    )


def print_results(localization_df: pd.DataFrame, curves_df: pd.DataFrame) -> None:
    """Print result tables to the console."""
    print("\nLocalization results")
    print("-" * 78)
    print(localization_df[["score", "precision_at_k", "recall_at_k", "iou"]].to_string(index=False))

    print("\nDeletion / insertion results")
    print("-" * 78)
    for score in curves_df["score"].unique():
        sub = curves_df[curves_df["score"] == score].copy()
        print(f"\nScore: {score}")
        print(sub.drop(columns=["display_name"], errors="ignore").to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def run_experiment(cfg: SyntheticConfig) -> Path:
    """Run the full synthetic evidence-localization experiment."""
    set_seed(cfg.seed)

    out_dir = make_output_dir(cfg)

    print("=" * 78)
    print("Synthetic Evidence Localization Experiment")
    print("=" * 78)
    print("Output directory:", out_dir)
    print("Device:", cfg.device)
    print("Seed:", cfg.seed)
    print("n_samples:", cfg.n_samples)
    print("length:", cfg.length)
    print("motif_len:", cfg.motif_len)
    print("noise_std:", cfg.noise_std)
    print("motif_amp:", cfg.motif_amp)
    print("add_distractor:", cfg.add_distractor)
    print("distractor_amp:", cfg.distractor_amp)
    print("lambda_sparse:", cfg.lambda_sparse)

    X, y, mask = generate_synthetic_dataset(
        n_samples=cfg.n_samples,
        length=cfg.length,
        motif_len=cfg.motif_len,
        noise_std=cfg.noise_std,
        motif_amp=cfg.motif_amp,
        add_distractor=cfg.add_distractor,
        distractor_amp=cfg.distractor_amp,
        distractor_len=cfg.distractor_len,
        seed=cfg.seed,
    )

    (
        X_train, y_train, _,
        X_val, y_val, _,
        X_test, y_test, mask_test,
    ) = train_val_test_split(
        X=X,
        y=y,
        mask=mask,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    train_loader, val_loader, test_loader = make_loaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    model = EvidenceGatedSSM(
        input_dim=1,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_classes=2,
    ).to(cfg.device)

    print("\nTraining model...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_device=cfg.device,
        lr=cfg.lr,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        grad_clip=cfg.grad_clip,
        lambda_sparse=cfg.lambda_sparse,
    )

    print("\nEvaluating classification...")
    accuracy, macro_f1 = evaluate_classification(model, test_loader, cfg.device)

    print("\nClassification results")
    print("-" * 78)
    print(f"Test accuracy : {accuracy:.4f}")
    print(f"Test macro-F1 : {macro_f1:.4f}")

    localization_df, n_positive = evaluate_localization(
        model=model,
        X_test=X_test,
        y_test=y_test,
        mask_test=mask_test,
        run_device=cfg.device,
        score_names=DEFAULT_SCORE_NAMES,
    )

    curves_df = evaluate_deletion_insertion(
        model=model,
        X_test=X_test,
        y_test=y_test,
        run_device=cfg.device,
        score_names=DEFAULT_SCORE_NAMES,
        ratios=cfg.evidence_ratios,
    )

    print_results(localization_df, curves_df)

    export_results(
        localization_df=localization_df,
        curves_df=curves_df,
        cfg=cfg,
        out_dir=out_dir,
        accuracy=accuracy,
        macro_f1=macro_f1,
        n_positive=n_positive,
    )

    print("\nSaved outputs in:", out_dir)
    print("Done.")

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic ground-truth evidence-localization experiment."
    )

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--output-dir", type=str, default="synthetic_results")

    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--motif-len", type=int, default=15)

    parser.add_argument("--noise-std", type=float, default=0.50)
    parser.add_argument("--motif-amp", type=float, default=2.0)

    parser.add_argument("--add-distractor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distractor-amp", type=float, default=2.5)
    parser.add_argument("--distractor-len", type=int, default=10)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=0.5)

    parser.add_argument("--lambda-sparse", type=float, default=0.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SyntheticConfig(
        seed=args.seed,
        device=resolve_device(args.device),
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples,
        length=args.length,
        motif_len=args.motif_len,
        noise_std=args.noise_std,
        motif_amp=args.motif_amp,
        add_distractor=args.add_distractor,
        distractor_amp=args.distractor_amp,
        distractor_len=args.distractor_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        lambda_sparse=args.lambda_sparse,
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()
