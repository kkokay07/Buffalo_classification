# Buffalo Breed Classification Pipeline

A complete computational pipeline for **buffalo breed classification** using genomic SNP data. The pipeline combines population-genetics analyses (QC, AIMs, ADMIXTURE, iHS) with machine learning classifiers to identify **Breed Informative Markers (BIMs)** and build a frozen, deployable classification model.

## Overview

The pipeline was applied to buffalo genomic data (96 individuals, 1,997 SNPs) from three breed groups (MIL – milk, DUA – dual-purpose, DRA – draft). The best individual classifiers (**SVM with linear kernel** and **MLP neural network**) achieved **93.75% test accuracy**. The final frozen model uses a small set of highly discriminative BIMs (AIMs ∪ iHS outliers).

**Pipeline flow:**

```
PLINK QC → AIM statistics → ADMIXTURE → Reference panel refinement
        → ML classifiers + ensembling → Incremental AIM classification
        → iHS analysis + union → BIM identification + model freezing
        → Final plots → (optional) XGBoost/NaiveBayes comparison
```

---

## Requirements

### Software

| Tool | Version | Purpose |
|------|---------|---------|
| [PLINK](https://www.cog-genomics.org/plink/) | v1.90b6.x | QC, genotype extraction |
| [ADMIXTURE](https://dalexander.github.io/admixture/) | v1.3.0 | Population structure analysis |
| Python | ≥ 3.10 | All Python scripts |
| Bash | any | Shell scripts |

### Python packages

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
matplotlib >= 3.7
seaborn >= 0.12
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Input data

- PLINK binary files (`2000ml.bed`, `2000ml.bim`, `2000ml.fam`) placed in the working directory. In the `.fam` file, the family ID (column 1) must encode the breed label (e.g., MIL, DUA, DRA).
- *(Optional, for step 8)* Standardized iHS scan output files, one per breed, named with the breed prefix, e.g.:
  `BHA_unstanderdized_iHS_file_3.ihs.out.100bins.norm`, `MUR_...`, `MEH_...`, `PAN_...`, `SUR_...` (tab-separated, standardized iHS in column 7).

> **Note:** Scripts use hard-coded relative paths and must be run **in order from the project root directory**, since each step consumes the output of the previous one.

---

## How to Use — Step by Step

### Step 1–2: PLINK quality control

```bash
bash 1_plink_qc.sh
```

Removes sex-chromosome (X, Y) and mitochondrial SNPs, TOD individuals, applies SNP call-rate (`--geno 0.05`) and individual missingness (`--mind 0.1`) filters.

**Output:** `qc_results/step4_mind.{bed,bim,fam}`

### Step 3: Calculate AIM statistics

```bash
python3 2_calculate_aims.py
```

Computes three ancestry-informativeness statistics per SNP:
- **Wright's FST** (Weir–Cockerham estimator)
- **Informativeness for Assignment (In)** (Rosenberg et al. 2002)
- **Delta** (Shriver et al. 2003)

Selects the top 500 SNPs per metric and by combined average rank.

**Output:** `aim_results/` (`aim_statistics.csv`, `top_aims_*.csv`, `top_aims_snplist.txt`)

### Step 4: ADMIXTURE analysis

```bash
bash 3_admixture_analysis.sh
```

Runs ADMIXTURE for K = 2 … (number of breeds + 2, max 7) with cross-validation (`--cv`).

**Output:** `admixture_results/` (`K*.Q`, `K*.P`, `cv_errors.txt`, `populations.txt`, `ind_order.txt`)

### Step 5: ADMIXTURE plots

```bash
python3 4_admixture_plots.py
```

Generates ancestry bar plots for every K and a cross-validation error plot, and prints mean ancestry proportions per breed.

**Output:** `admixture_plots/admixture_K*.png`, `admixture_plots/cv_errors.png`

### Step 6: Refine the reference panel

```bash
python3 5_refine_reference_panel.py
```

Applies the **Crum et al. (2019)** criterion: retains only individuals with ≥ 80% ancestry in their breed's dominant ADMIXTURE component (uses `K5.Q`; adjust in the script if a different K is preferred).

**Output:** `refined_panel/refined_ref_panel.{bed,bim,fam}`, `keep_individuals.txt`, `removed_individuals.csv`, `refinement_summary.csv`

### Step 7: ML classifiers and ensembling

```bash
python3 6_ml_classifiers.py
```

Trains 11 classifiers (RandomForest, ExtraTrees, GradientBoosting, AdaBoost, LogisticRegression, SVM-RBF, SVM-Linear, KNN, NaiveBayes, DecisionTree, MLP) on the refined panel with an 80/20 stratified train/test split and 5-fold cross-validation. Evaluates accuracy, precision, recall, specificity, F1, MCC and AUC, then builds four ensemble methods (hard voting, soft voting, stacking with logistic regression, weighted average) from the top-3 models.

**Output:** `ml_results/classifier_results.csv`, `ensemble_results.csv`, `all_results.csv`

### Step 8: Incremental AIM classification

```bash
python3 7_aims_incremental_classification.py
```

Classifies breeds using the top 10, 20, 30, …, 200 AIMs (by combined rank) with SVM-Linear and MLP, to find the optimal AIM panel size. Saves the best-performing AIM lists.

**Output:** `incremental_aims/incremental_classification_results.csv`, `best_aims_svm.txt`, `best_aims_mlp.txt`

### Step 9: iHS analysis and union with AIMs

```bash
python3 8_ihs_analysis_and_union.py
```

Reads all `*_unstanderdized_iHS_file_3.ihs.out.100bins.norm` files in the working directory, extracts positions with |standardized iHS| > 3 SD, maps them to SNPs, and takes the **union of best AIMs + iHS outlier SNPs**. Re-runs the full classifier + ensemble comparison on this reduced marker set.

**Output:** `ihs_union_results/` (union SNP lists, `classification_results_union.csv`, best-model info)

### Step 10: Identify BIMs and freeze the final model

```bash
python3 9_identify_bims_and_freeze_model.py
```

Ranks SNPs from the AIM ∪ iHS union by Random-Forest importance and breed allele-frequency differences, selects the top 10/20 **Breed Informative Markers (BIMs)**, and freezes the best model (soft-voting ensemble + scaler + label encoder) as a pickle file with a JSON metadata file for prediction on new samples.

**Output:** `final_model/` — `frozen_model.pkl`, `model_info.json`, `top_10_bims.txt`, `top_20_bims.txt`, `top_20_bims_detailed.csv`, `snp_importance.csv`, `breed_discrimination.csv`, `bim_genotypes_all_individuals.csv`

### Step 11: Final visualization plots

```bash
python3 10_create_final_plots.py
```

Generates publication-quality figures: BIM allele-frequency heatmap, accuracy vs. number of AIMs, classifier comparison, and BIM score plots.

**Output:** `final_model/bim_heatmap.png`, `final_model/bim_scores.png`, `incremental_aims/accuracy_by_n_aims.png`, `ihs_union_results/classifier_comparison.png`

### Step 12 (optional): XGBoost / Naive Bayes comparison

```bash
python3 11_xgboost_naivebayes_analysis.py
```

Benchmarks Gradient Boosting (scikit-learn XGBoost equivalent) and Gaussian Naive Bayes across varying SNP panel sizes, including a comparison against the full SNP set.

**Output:** `xgboost_naivebayes_results/xgboost_naivebayes_results.csv`, SNP importance rankings, confusion matrices

---

## Repository Structure

```
├── 1_plink_qc.sh                     # Step 1–2: QC
├── 2_calculate_aims.py               # Step 3: AIM statistics (FST, In, Delta)
├── 3_admixture_analysis.sh           # Step 4: ADMIXTURE
├── 4_admixture_plots.py              # Step 5: ADMIXTURE plots
├── 5_refine_reference_panel.py       # Step 6: Reference panel refinement
├── 6_ml_classifiers.py               # Step 7: 11 classifiers + 4 ensembles
├── 7_aims_incremental_classification.py  # Step 8: Incremental AIM panels
├── 8_ihs_analysis_and_union.py       # Step 9: iHS outliers ∪ AIMs
├── 9_identify_bims_and_freeze_model.py   # Step 10: BIMs + frozen model
├── 10_create_final_plots.py          # Step 11: Final figures
├── 11_xgboost_naivebayes_analysis.py # Step 12 (optional): GBM/NB benchmark
├── qc_results/                       # QC output
├── aim_results/                      # AIM statistics
├── admixture_results/                # ADMIXTURE output
├── admixture_plots/                  # ADMIXTURE figures
├── refined_panel/                    # Refined reference panel
├── ml_results/                       # Classifier & ensemble results
├── incremental_aims/                 # Incremental AIM results
├── ihs_union_results/                # iHS ∪ AIM results
├── final_model/                      # Frozen model + BIM lists
└── xgboost_naivebayes_results/       # GBM/NB benchmark results
```

---

## Key Results (from the study)

- **Best classifiers:** SVM (linear kernel) and MLP:  **93.75% test accuracy**, F1 = 0.933, MCC = 0.842, AUC = 1.000
- **Population structure (ADMIXTURE):** best K = 5 by CV error; DRA is highly distinct (95.2% unique ancestry)
- **Reference panel:** 78/96 individuals retained after applying the ≥ 80% ancestry criterion (18 admixed individuals removed)
- **BIMs:** a compact panel of top-ranked AIM ∪ iHS SNPs sufficient for near-perfect breed assignment

## Authors

[Rangasai Chandra Goli](https://scholar.google.com/citations?user=e-_I-e4AAAAJ&hl=en) |
[Kanaka KK](https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en) |
[Madhusudan Reddy Nandineni](https://cdfd.org.in/research-details/14/dr-madhusudan-reddy-nandineni)

## Citation

Currently the paper is under peer review in a reputed journal. Citation will be added once the paper get published.

## License

MIT license.
