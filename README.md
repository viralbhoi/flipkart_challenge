# 🚕 Spatio-Temporal Demand Forecasting Pipeline

## 📖 Overview

This project predicts geographic demand using spatio-temporal data (Location/Geohash, Time, Weather, etc.). Because of the size of the data and the complexity of the algorithms, we are utilizing a **Distributed 4-PC Pipeline**.

One "Master PC" handles all data cleaning and preprocessing to ensure absolute consistency. The processed data is then distributed to 4 separate computers, each training a different state-of-the-art machine learning model in parallel. Finally, the predictions are ensembled together for maximum accuracy.

## 🛠️ Prerequisites

Before starting, ensure all PCs have Python installed along with the required libraries.
Run this command in your terminal:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost joblib

```

---

## 🚀 The Workflow (Step-by-Step)

### Phase 1: Data Preprocessing (Master PC Only)

_Stop! Do not process data on individual machines. All data must be processed centrally to prevent column mismatches._

1. Place `train.csv` and `test.csv` in the root folder.
2. Run the two pipeline scripts:

```bash
python pipelineA.py
python pipelineB.py

```

3. **Output:** You will now have 4 new CSV files in your folder:

- `Train_Standard_Models.csv` & `Test_Standard_Models.csv` _(One-Hot Encoded)_
- `Train_CatBoost.csv` & `Test_CatBoost.csv` _(Raw Text / Target Encoded)_

### Phase 2: Parallel Training (The 4 PCs)

Distribute the generated CSV files to your team via USB, network drive, or cloud storage. Assign each team member one of the following models to train on their respective PC:

- **💻 PC 1 (Random Forest):** \* **Uses:** `Train_Standard_Models.csv`
- **Role:** Establishes a strong, stable baseline.

- **💻 PC 2 (XGBoost):** \* **Uses:** `Train_Standard_Models.csv`
- **Role:** The classic heavyweight gradient booster.

- **💻 PC 3 (LightGBM):** \* **Uses:** `Train_Standard_Models.csv`
- **Role:** Fast, leaf-wise tree growth optimized for skewed demand.

- **💻 PC 4 (CatBoost):** \* **Uses:** `Train_CatBoost.csv` (Notice it uses the different dataset!)
- **Role:** The secret weapon. Handles the high-cardinality `geohash` locations mathematically without exploding the column count.

**Instructions for Team Members:**
Run your assigned Python training script. When it finishes, it will output a predictions file (e.g., `preds_xgb.csv`).

### Phase 3: The Reunion (Master PC)

1. Collect the 4 prediction CSV files from your team members (`preds_rf.csv`, `preds_xgb.csv`, `preds_lgb.csv`, `preds_cat.csv`).
2. Place them back onto the Master PC in the same folder.
3. Run the final Ensembling script to average the predictions together.
4. **Final Output:** The script will generate `FINAL_ENSEMBLE_SUBMISSION.csv`. This is your final, highly accurate file ready for scoring/submission!

---

## ⚠️ Important Notes

- **The "Inverse" Phenomenon:** Based on EDA, demand peaks at 11:00 AM and crashes at 7:00 PM. This indicates the target variable likely represents _idle time_ or _availability_, not consumer volume. The models will handle this natively, but keep this in mind for stakeholder presentations.
