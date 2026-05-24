# Evidence-Gated Additive State-Space Classifier

This repository contains the code used for the experiments in the manuscript on **unnormalized gated evidence accumulation for interpretable time-series classification**.

The main purpose of the repository is reproducibility. The code is organized around the experimental sections of the paper: synthetic ground-truth evidence localization, UCR aggregation ablation, evidence-level deletion curves, classifier-aware deletion/insertion, qualitative evidence profiles, and perturbed-input evaluation.

---

## 1. Repository structure

```text
evidence-gated-ssm/
│
├── README.md
├── requirements.txt
│
├── process_datasets.py
│
├── run_synthetic_evidence_localization.py
├── run_main_accuracy_multiseed_aggregation_ablation.py
├── run_paired_statistical_tests.py
├── run_evidence_deletion_curves.py
├── run_ucr_score_deletion_insertion.py
├── run_qualitative_evidence_profiles.py
└── run_perturbed_input_evaluation.py
```

Recommended local data/output layout:

```text
evidence-gated-ssm/
│
├── Datasets/
│   ├── ECG5000/
│   │   ├── ECG5000_TRAIN
│   │   └── ECG5000_TEST
│   ├── Wafer/
│   │   ├── Wafer_TRAIN
│   │   └── Wafer_TEST
│   └── ...
│
└── Results/
    ├── *.csv
    ├── *.pdf
    └── *.png
```

The dataset loader also supports a flat UCR layout such as:

```text
Datasets/ECG5000_TRAIN
Datasets/ECG5000_TEST
```

---

## 2. Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
numpy
pandas
scipy
scikit-learn
torch
matplotlib
```

The scripts use PyTorch and automatically use CUDA when available if `--device auto` is selected.

---

## 3. Datasets

The UCR experiments use the following ten selected datasets:

```text
ECG5000
Wafer
ElectricDevices
FaceAll
PhalangesOutlinesCorrect
CricketX
SwedishLeaf
UWaveGestureLibraryX
Yoga
Earthquakes
```

Download the corresponding UCR `.TRAIN` and `.TEST` files from the UCR Time Series Classification Archive and place them under `Datasets/`.

Both of the following layouts are accepted:

```text
Datasets/<DatasetName>/<DatasetName>_TRAIN
Datasets/<DatasetName>/<DatasetName>_TEST
```

or:

```text
Datasets/<DatasetName>_TRAIN
Datasets/<DatasetName>_TEST
```

The loader independently standardizes each time series:

`x = (x - mean(x)) / std(x)`

Labels are one-hot encoded using the union of train and test labels, then converted to integer class indices inside the experiment scripts.

---

## 4. Reproducing the experiments

All UCR experiments use the five random seeds:

```text
2025, 2026, 2027, 2028, 2029
```

The default training configuration is:

```text
hidden_dim = 64
latent_dim = 64
learning rate = 5e-4
batch size = 64
maximum epochs = 50
early stopping patience = 10
gradient clipping max norm = 0.5
validation split = 20% of the original training set
```

### 4.1 Synthetic ground-truth evidence localization

Corresponds to the synthetic localization experiment.

```bash
python run_synthetic_evidence_localization.py --device auto
```

Main outputs:

```text
synthetic_results/
└── synthetic_distractor_lambda_0_seed_2025/
    ├── experiment_config.txt
    ├── localization_results.csv
    ├── deletion_insertion_curves.csv
    ├── localization_iou.png
    ├── localization_iou.pdf
    ├── deletion_accuracy_curves.png
    ├── deletion_accuracy_curves.pdf
    ├── deletion_macro_f1_curves.png
    ├── deletion_macro_f1_curves.pdf
    ├── insertion_accuracy_curves.png
    ├── insertion_accuracy_curves.pdf
    ├── insertion_macro_f1_curves.png
    └── insertion_macro_f1_curves.pdf
```

### 4.2 Main UCR aggregation ablation

Corresponds to the main classification tables comparing:

```text
PlainSSM
NormGated
Proposed_Unnormalized_Base

```

Run:

```bash
python run_main_accuracy_multiseed_aggregation_ablation.py --device auto --results-dir Results
```

Main outputs:

```text
Results/multiseed_aggregation_ablation_detail_<timestamp>.csv
Results/multiseed_aggregation_ablation_summary_<timestamp>.csv
```

The detail CSV is used as input for the statistical comparison script.

### 4.3 Dataset-level paired statistical comparison

Corresponds to the dataset-level paired comparison table.

Run this after the main aggregation-ablation script:

```bash
python run_paired_statistical_tests.py \
  --input-detail-csv Results/multiseed_aggregation_ablation_detail_<timestamp>.csv \
  --results-dir Results
```

Main outputs:

```text
Results/dataset_level_paired_statistical_tests_summary.csv
Results/dataset_level_paired_statistical_tests_dataset_details.csv
Results/dataset_level_paired_statistical_tests_table.tex
```

The statistical tests are computed on dataset-level means over seeds, avoiding a pooled dataset--seed independence assumption.

### 4.4 Evidence-level deletion curves

Corresponds to the evidence-norm deletion-curve experiment.

```bash
python run_evidence_deletion_curves.py --device auto --results-dir Results
```

This script uses:

$$
s_t^{(\mathrm{norm})} = \|e_t\|_2 = \|\alpha_t z_t\|_2.
$$

It compares:

```text
high-evidence deletion
random deletion
low-evidence deletion
```

at deletion ratios:

```text
10%, 20%, 30%, 40%, 50%
```

Main outputs:

```text
Results/evidence_deletion_curves_detail_<timestamp>.csv
Results/evidence_deletion_curves_summary_<timestamp>.csv
Results/evidence_deletion_curves_average_<timestamp>.pdf
Results/evidence_deletion_curves_average_<timestamp>.png
```

### 4.5 Classifier-aware deletion and insertion

Corresponds to the classifier-aware score comparison.

```bash
python run_ucr_score_deletion_insertion.py --device auto --results-dir Results
```

The script compares six temporal evidence scores:

```text
random
latent_norm
gate_alpha
evidence_norm
class_logit
margin_logit
```

with both:

```text
deletion
insertion
```

at the 10% evidence ratio.

Main outputs:

```text
Results/ucr_score_deletion_insertion_detail_<timestamp>.csv
Results/ucr_score_deletion_insertion_summary_<timestamp>.csv
Results/ucr_score_deletion_insertion_overall_<timestamp>.csv
```

### 4.6 Qualitative evidence profiles

Corresponds to the qualitative evidence-profile figure.

```bash
python run_qualitative_evidence_profiles.py --device auto --results-dir Results
```

By default, this script exports representative profiles for:

```text
ECG5000
ElectricDevices
```

Main outputs:

```text
Results/qualitative_evidence_profiles_<timestamp>/
├── qualitative_evidence_profile_<dataset>_idx<...>.pdf
├── qualitative_evidence_profile_<dataset>_idx<...>.png
└── selected_qualitative_profiles.csv
```

### 4.7 Perturbed-input evaluation

Corresponds to the perturbed-input robustness/evaluation experiment.

```bash
python run_perturbed_input_evaluation.py --device auto --results-dir Results
```

The script compares:

```text
PlainSSM
Proposed_Unnormalized_Base
```

under:

```text
clean
Gaussian noise
random masking
local interval corruption
```

Default perturbation levels:

```text
Gaussian noise sigma: 0.05, 0.10, 0.20
Random masking ratio: 0.10, 0.20, 0.30
Local interval corruption ratio: 0.10, 0.20, 0.30
```

Main outputs:

```text
Results/perturbed_input_evaluation_detail_<timestamp>.csv
Results/perturbed_input_evaluation_summary_<timestamp>.csv
Results/perturbed_input_evaluation_overall_<timestamp>.csv
```

---

## 5. Output files and manuscript tables

The following mapping summarizes which scripts reproduce which manuscript components.

| Manuscript component | Script | Main output |
|---|---|---|
| Synthetic localization | `run_synthetic_evidence_localization.py` | `localization_results.csv`, `deletion_insertion_curves.csv` |
| Main UCR classification | `run_main_accuracy_multiseed_aggregation_ablation.py` | `multiseed_aggregation_ablation_summary_*.csv` |
| Dataset-level statistical comparison | `run_paired_statistical_tests.py` | `dataset_level_paired_statistical_tests_table.tex` |
| Evidence deletion curves | `run_evidence_deletion_curves.py` | `evidence_deletion_curves_summary_*.csv`, average curve figure |
| Classifier-aware deletion/insertion | `run_ucr_score_deletion_insertion.py` | `ucr_score_deletion_insertion_overall_*.csv` |
| Qualitative profiles | `run_qualitative_evidence_profiles.py` | qualitative profile PDF/PNG files |
| Perturbed-input evaluation | `run_perturbed_input_evaluation.py` | `perturbed_input_evaluation_overall_*.csv` |

---

## 6. Notes on reproducibility

The scripts set NumPy and PyTorch seeds before each run. Some small numerical variation may still occur across hardware, CUDA versions, and PyTorch versions.

The experiments use early stopping based on validation macro-F1 for UCR experiments, except the synthetic experiment, which uses validation loss. The UCR test split is used only for final evaluation.

The deletion and insertion experiments are representation-level analyses: the hidden trajectory is computed from the original input sequence, and then selected additive evidence terms are removed or retained in the final aggregation. The perturbed-input evaluation is different: it modifies the input sequence and recomputes the full model prediction.

---

## 7. Citation

If you use this repository, please cite the associated manuscript.

```bibtex
@article{your_citation_key,
  title   = {Evidence-Gated State-Space Networks for Interpretable Time Series Classification},
  author  = {Aykut T. Altay},
  journal = {To appear},
  year    = {2026}
}
```
