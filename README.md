# The Performance–Stability Gap: Are SHAP Feature Rankings Robust to Train–Test Split Ratios?

Code accompanying the manuscript **"Are SHAP Feature Rankings Robust to Train–Test Split Ratios? A Repeated-Split Study Using Random Forest on Two Clinical Datasets"** (submitted to *Health Information Science and Systems*).

The study asks whether the train–test split ratio affects (a) Random Forest predictive performance and (b) the stability of SHAP-based feature-importance rankings, and whether the best-performing ratio is also the one with the most stable explanations.

## Repository contents

| File | Description |
| --- | --- |
| `Final_diabetes.ipynb` | Experiment 1 — Pima Indians Diabetes dataset |
| `Heart_Disease_final.ipynb` | Experiment 2 — combined UCI heart-disease dataset |
| `diabetes.csv` | Pima Indians Diabetes data (768 rows, 8 predictors + `Outcome`) |
| `heart_disease_combined.csv` | Heart-disease data as downloaded (2,943 rows, 10 predictors + `HeartDisease`) |
| `requirements.txt` | Python dependencies (versions as in the manuscript, Table 4) |

## Study design (both experiments)

- **Model:** Random Forest (scikit-learn) — the only algorithm used.
- **Split ratios (train:test):** 50:50, 60:40, 70:30, 80:20, 90:10, all stratified.
- **Repeats:** 20 matched seeds (0–19) per ratio, so 100 runs per dataset. Seed *k* drives the *k*-th split under every ratio, which allows a paired Friedman test.
- **Leakage-safe pipeline:** imputation → IQR outlier capping (winsorising, never dropping rows) → scaling / one-hot encoding → Random Forest, all fitted on training data only.
- **Imputation:** MICE (`IterativeImputer` + `BayesianRidge`) for Pima, because of heavy missingness (Insulin ≈ 48.7 %, SkinThickness ≈ 29.6 %); median for the heart data.
- **Class imbalance:** `class_weight="balanced"`; no SMOTE or other resampling.
- **Tuning:** `RandomizedSearchCV` (25 iterations, stratified 5-fold CV, scored on PR-AUC), re-run independently for every (seed, ratio) pair.
- **Performance:** Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC (mean ± 95 % CI over 20 seeds); Friedman test on PR-AUC with Nemenyi post-hoc comparisons when significant.
- **Explainability:** SHAP `TreeExplainer` on the held-out test set; features ranked by mean |SHAP|. Stability across the 10 ratio pairs is measured with Kendall's τ, Spearman's ρ, Jaccard overlap (Top-3 and Top-5) and mean absolute rank difference, computed on mean ranks averaged over the 20 seeds. For the heart data, one-hot columns are summed back to their parent feature.

### Dataset-specific notes

- **Pima:** zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are treated as missing.
- **Heart:** zeros in `Cholesterol` and `RestingBP` are treated as missing. 723 of 2,943 rows (24.6 %) are exact duplicates and are dropped, leaving 2,220 rows (56.4 % positive).

## Data sources

- Pima Indians Diabetes: <https://www.kaggle.com/datasets/mragpavank/diabetes> (originally from the National Institute of Diabetes and Digestive and Kidney Diseases; Smith et al., 1988).
- Heart disease: <https://www.kaggle.com/datasets/rcratos/heart-disease-data-compiled-from-uci/data>.

Please consult the original dataset pages for their licence and citation requirements.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open a notebook and use **Kernel → Restart & Run All**. Keep the CSV files in the same folder as the notebooks (they are loaded by relative path).

### Runtime — please read

Each notebook's main loop (Section 8) performs 5 ratios × 20 seeds × 25 configurations × 5 folds = **12,500 Random Forest fits** for tuning, plus 100 final refits and 100 SHAP computations. Expect this to take **hours** on a laptop (the manuscript's runs used an 8-logical-core, 16 GB RAM machine). To smoke-test quickly, lower `N_REPEATS`, `N_ITER` and `CV_FOLDS` in Section 7 — but note that results will then differ from those reported.

### Outputs

Running the notebooks writes publication-style figures to the working directory (`Fig1`–`Fig2` for Pima, `Fig3`–`Fig4` for heart, each as `.pdf`, `.eps` and 600-dpi `.tiff`). Note that these figure numbers are the notebooks' own file names and do not correspond one-to-one with the figure numbers in the manuscript. The saved notebooks already contain the outputs from the reported run.

## Reproducibility notes

- Seeds 0–19 are fixed for the splits, CV folds and hyperparameter search; the Random Forest and imputer use `random_state=42`.
- Exact numbers can shift slightly if library versions differ from `requirements.txt`, particularly `scikit-learn` and `shap`.
- SHAP values are interpreted as associations learned by the model, not causal effects.

## Citation

If you use this code, please cite the manuscript (full reference to be added once published).

## Licence

No licence has been specified yet. Add a `LICENSE` file (e.g. MIT) before making the repository public if you want others to be able to reuse the code.