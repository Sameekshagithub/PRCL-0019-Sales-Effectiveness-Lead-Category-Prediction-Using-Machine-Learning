<div align="center">

# 📊 Sales Effectiveness — Lead Category Prediction

**Predicting High-Potential vs. Low-Potential sales leads for FicZon Inc. using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas)](https://pandas.pydata.org/)
[![MySQL](https://img.shields.io/badge/MySQL-Data%20Source-4479A1?logo=mysql)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

</div>

---

## 📌 Overview

FicZon Inc. is an IT solutions provider whose leads are generated primarily through digital channels. Lead quality categorization is currently a **manual, post-hoc process** carried out by sales staff — valuable for analysis, but too late to influence real-time conversion decisions.

This project builds a **binary classification pipeline** that pre-categorizes each incoming lead as:

- 🟢 **High Potential** — real sales progress / conversion signal
- 🔴 **Low Potential** — no confirmed sales progress

...at the point of capture, so sales agents can prioritize effort on the leads most likely to convert.

| | |
|---|---|
| **Client** | FicZon Inc. (IT Solutions Provider) |
| **Dataset** | 7,422 records · 9 columns (MySQL: `project_sales.data`) |
| **Task** | Binary Classification (High Potential / Low Potential) |
| **Team Code** | PTID-AIE-JUL-26-11142 |
| **Project Code** | PM-PR-0019 |

---

## 🎯 Business Objective

1. Generate data exploration insights related to sales effectiveness.
2. Build a Machine Learning classifier for `Lead_Category` so sales agents can focus effort on the leads most likely to convert — shifting lead quality assessment from **post-analysis** to **real-time, point-of-capture scoring**.

---

## 🗂️ Dataset

| Column | Description |
|---|---|
| `Product_ID` | Identifier for the product/service line associated with the lead |
| `Source` | Channel through which the lead was generated (Website, Referral, Ads, etc.) |
| `Mobile` | Lead's contact number (used only to derive a "has contact info" flag) |
| `EMAIL` | Lead's email address (same as above) |
| `Sales_Agent` | Agent assigned to the lead |
| `Location` | Geographic location of the lead |
| `Delivery_Mode` | On-premises vs. SaaS delivery preference |
| `Created` | Timestamp the lead was captured |
| `Status` | Raw pipeline status — the field `Lead_Category` is derived from |

> ⚠️ **Note:** `Lead_Category` is **not** a raw column — it is engineered from `Status` (see [Target Engineering](#-target-engineering--a-critical-gotcha)).

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Access | `mysql-connector-python`, `getpass` (secure credential entry) |
| Data Wrangling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Modeling | `scikit-learn` |
| Environment | Jupyter Notebook |

---

## 🏗️ Project Pipeline

```
MySQL DB ──▶ Load & Basic Checks ──▶ EDA ──▶ Data Cleaning
   ──▶ Target Engineering (Status → Lead_Category)
   ──▶ Feature Engineering ──▶ Encoding & Scaling
   ──▶ Model Training (5 algorithms) ──▶ Overfitting Diagnostics
   ──▶ Hyperparameter Tuning (GridSearchCV) ──▶ Model Comparison
   ──▶ Feature Importance ──▶ Business Insights
```

### 1. Data Cleaning
- **Nulls:** `Source`, `Sales_Agent`, `Location`, `Mobile` → imputed with an explicit `'Unknown'` category (missingness kept as a usable signal rather than discarded); `Product_ID` → median imputation.
- **Duplicates:** exact duplicate rows dropped.
- **Outliers:** IQR-based detection on `Product_ID` (the only continuous numeric field), values winsorized (capped) at the IQR fences rather than dropped, so no leads are lost.

### 2. Target Engineering — a critical gotcha ⚠️
The raw table has no `Lead_Category` column — it's derived from `Status` via an explicit mapping. This step **must** be built from the *exact, verified* values returned by `df['Status'].unique()`, not assumed placeholder labels — a mismatch here (e.g. `'Converted'` vs. the real `'CONVERTED'`/`'converted'`) can silently collapse the entire target into a single class, which breaks Logistic Regression / Gradient Boosting training outright and produces meaningless 100% "accuracy" from other models. Always re-check `df['Lead_Category'].value_counts()` shows a genuine multi-class split before modeling.

### 3. Feature Engineering
- `Created` → `Created_Year`, `Created_Month`, `Created_Day` (seasonality signal)
- `Mobile`, `EMAIL` → binary flags `Has_Mobile`, `Has_Email`
- `Status` dropped post-derivation to prevent target leakage

### 4. Encoding & Scaling
- One-hot encoding for `Source`, `Sales_Agent`, `Location`, `Delivery_Mode`
- `StandardScaler` applied for scale-sensitive models (Logistic Regression, KNN); tree-based models use unscaled features

---

## 🤖 Models Trained

| Model | Key Regularization Applied |
|---|---|
| Logistic Regression | `C=0.5` (L2 penalty), `class_weight='balanced'` |
| Decision Tree | `max_depth`, `min_samples_leaf`, `min_samples_split`, `class_weight='balanced'` |
| Random Forest | `max_depth`, `min_samples_leaf`, `class_weight='balanced'` |
| K-Nearest Neighbors | Tuned `n_neighbors` for smoother decision boundary |
| Gradient Boosting | Low `learning_rate`, `subsample < 1` (stochastic boosting), shallow `max_depth` |
| **Random Forest (Tuned)** | `GridSearchCV` over depth/estimators/leaf-size, `scoring='f1_weighted'` |

### Overfitting Diagnostics
Every model is evaluated on **Train Accuracy**, **5-fold Stratified CV Accuracy**, and **Test Accuracy**, with the Train–Test gap reported explicitly — not just a single test-set number — to make overfitting visible and measurable rather than assumed away.

---

## 📈 Evaluation

Models are compared using **Accuracy, Precision, Recall, and F1-Score** (weighted, for class imbalance), plus a confusion matrix and feature importance ranking (via Random Forest) for the best-performing model.

---

## 💡 Business Insights

- **Data-Driven Prioritization:** Sales agents can see which incoming leads are High Potential before manual review.
- **Channel Effectiveness:** `Source` and `Delivery_Mode` carry meaningful predictive weight — some acquisition channels consistently produce higher-quality leads.
- **Agent & Location Patterns:** Regional demand and agent handling both influence conversion likelihood.
- **Manual → ML-Assisted:** Moves lead-quality assessment from post-hoc analysis into the live sales workflow.

---

## 🚧 Challenges Faced

1. No ready-made target column — required deriving and *verifying* the `Status → Lead_Category` mapping against real data values.
2. Missing values spread across multiple columns, each needing column-appropriate treatment.
3. High-cardinality categorical features (`Sales_Agent`, `Location`) inflate the one-hot feature space.
4. Secure handling of remote MySQL credentials (`getpass`, never hard-coded).

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/sales-effectiveness-lead-prediction.git
cd sales-effectiveness-lead-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
numpy
matplotlib
seaborn
scikit-learn
mysql-connector-python
jupyter
```

---

## ▶️ Usage

```bash
jupyter notebook PRCL_0019_Sales_Effectiveness_CORRECTED.ipynb
```

You will be prompted at runtime for the database password via `getpass()` — credentials are **never** hard-coded or stored in the notebook.

```python
DB_HOST = "your-db-host"
DB_PORT = 3306
DB_NAME = "project_sales"
DB_USER = "your-username"
DB_PASSWORD = getpass("Enter database password: ")
```

Run all cells sequentially — the pipeline flows from data load → cleaning → target engineering → modeling → evaluation.

---

## 📁 Repository Structure

```
├── PRCL_0019_Sales_Effectiveness_CORRECTED.ipynb   # Main analysis & modeling notebook
├── requirements.txt                                 # Python dependencies
├── README.md                                        # Project documentation
└── /assets                                          # (optional) exported charts/plots
```

---

## 🔭 Next Steps

- Validate the `Status → Lead_Category` mapping with FicZon's sales team.
- Monitor model performance on new leads over time; retrain periodically as lead patterns evolve.
- Explore grouping/target-encoding for high-cardinality `Sales_Agent` and `Location` fields.
- Deploy the tuned model behind a lightweight scoring API for real-time lead capture.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 Author

**Submitted by:** Sameeksha
**Category:** Product Sales · Data Science Mentoring Project

<div align="center">

⭐ If this project was useful, consider giving it a star!

</div>
