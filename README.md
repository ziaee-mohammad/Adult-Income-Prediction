# 👔 Adult Income Prediction (Census)

A clean, reproducible **tabular ML** project to predict whether a person’s annual income is **>50K** using the **Adult Census Income** dataset.  
Includes preprocessing with `ColumnTransformer`, feature engineering, model training (Logistic Regression, Random Forest, XGBoost), **threshold tuning**, and rigorous evaluation (ROC‑AUC, PR‑AUC, F1).

---

## 📖 Overview
This repository demonstrates best practices for **tabular classification** with mixed numeric/categorical features.  
The pipeline prevents leakage by fitting all transforms **inside** `sklearn.Pipeline`, supports cross‑validation, and reports metrics robust to **class imbalance**.

**Highlights**
- One‑file reproducible pipeline (EDA → features → training → evaluation)  
- `ColumnTransformer` for imputation, scaling, and one‑hot encoding  
- Hyperparameter tuning with cross‑validation  
- Threshold selection aligned with business goals (F1 / F2 / Youden‑J)  
- Model explainability (feature importance, optional SHAP)

---

## 🗂️ Dataset
- **Source**: UCI Adult/Census Income dataset (replace with your link if local copy is used).  
- **Target**: `income` (`>50K` vs `<=50K`).  
- **Typical features** (`13`): `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.

Example layout:
```
data/
├─ adult_train.csv
└─ adult_test.csv
```

---

## 🔧 Preprocessing & Feature Engineering
- **Imputation**: numeric (median), categorical (most frequent)  
- **Scaling**: StandardScaler for numeric columns  
- **Encoding**: One‑hot for categoricals (`handle_unknown="ignore"`)  
- **Domain features (optional)**: work hours buckets, capital gain/loss flags, interaction terms  
- **Class imbalance**: class weights on models; report PR‑AUC

All steps live **inside** `Pipeline` to avoid leakage.

---

## 🧠 Models
- **Logistic Regression** — baseline + interpretability  
- **Random Forest** — non‑linear baseline  
- **XGBoost** — strong tabular learner (supports class weights)  
- *(Optional)* LightGBM / CatBoost

---

## 📈 Evaluation
Primary metrics:
- **ROC‑AUC** and **PR‑AUC** (important if classes are imbalanced)  
- **F1 / F2** and **Accuracy** for a chosen threshold  
- **Confusion matrix** at the operating point

**Threshold selection**:
- Sweep thresholds and pick by **F1** (balanced) or **F2** (recall‑oriented) or **Youden‑J** (sensitivity + specificity).

---

## 🧩 Repository Structure (suggested)
```
Adult-Income-Prediction/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_training_baselines.ipynb
│  ├─ 03_threshold_selection.ipynb
├─ src/
│  ├─ data.py          # load/split; leakage-safe transforms
│  ├─ features.py      # column definitions & preprocessors
│  ├─ models.py        # model builders (logreg/RF/XGB)
│  ├─ train.py         # fit + CV
│  ├─ eval.py          # metrics, curves, threshold sweep
│  └─ explain.py       # SHAP / feature importance (optional)
├─ reports/figures/    # ROC/PR curves, CM, importances
├─ data/               # (gitignored)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Setup & Usage
1) **Clone & install**
```bash
git clone https://github.com/ziaee-mohammad/Adult-Income-Prediction.git
cd Adult-Income-Prediction
pip install -r requirements.txt
```

2) **Run notebooks**
```bash
jupyter notebook
```

3) **Or use scripts (if added)**
```bash
python -m src.train --model xgboost
python -m src.eval  --threshold auto_f1
```

---

## 📦 Requirements (example)
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
shap           # optional
```

---

## ✅ Fairness & Responsible ML (recommended)
- Report metrics **by subgroup** (e.g., `sex`, `race`, `age` buckets) to monitor disparate impact.  
- Avoid using **sensitive attributes** directly in the final model unless policy allows and you’ve run bias assessments.  
- Consider **thresholds per segment** only if policy-compliant and justified.

---

## 📈 Example Results (replace with your numbers)
| Model | ROC‑AUC | PR‑AUC | F1 | Accuracy |
|------|--------:|------:|---:|--------:|
| Logistic Regression | 0.89 | 0.70 | 0.78 | 0.84 |
| Random Forest | 0.91 | 0.73 | 0.80 | 0.86 |
| **XGBoost** | **0.93** | **0.76** | **0.82** | **0.87** |

> Include confusion matrix at the selected threshold and a short feature-importance chart.

---

## 🏷 Tags
```
data-science
machine-learning
classification
tabular-data
feature-engineering
model-evaluation
xgboost
python
jupyter-notebook
```

---

## 👤 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad

---

## 📜 License
MIT — free to use and adapt with attribution.
