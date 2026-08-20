# 🛡️ NetGuard AI

**AI-Powered Network Traffic Anomaly Detection**

A simple, professional Streamlit dashboard built around a pre-trained network
traffic anomaly detection system. NetGuard AI never trains models — it only
loads existing artifacts and uses them for inference.

---

## 1. Project Overview

NetGuard AI lets you:

- Enter a single network flow's features and classify it as **BENIGN** or
  **ATTACK** using Random Forest, XGBoost, or LSTM.
- Run unsupervised anomaly detection with **Isolation Forest** or
  **Autoencoder**.
- Classify the **attack type** with a multiclass Random Forest model.
- Upload a CSV of many traffic records for **batch detection**.
- Compare model evaluation metrics (if available).
- Explore the training dataset's class balance and feature distributions
  (if a dataset is provided).

The app is intentionally simple: one `app.py` file, six sidebar pages, and no
hidden magic. It is built for a portfolio project, a university
demonstration, a supervisor meeting, or a job interview — while still being
easy for a student to read and maintain.

---

## 2. Required Model Files

Place your already-trained artifacts inside the `models/` folder, using
**exactly these filenames**:

```text
models/
├── scaler_binary.pkl
├── scaler_multi.pkl
├── label_encoder.pkl
├── random_forest_binary.pkl
├── random_forest_multiclass.pkl
├── xgboost_binary.pkl
├── isolation_forest.pkl
├── lstm_model.keras            (optional)
└── autoencoder_model.keras     (optional)
```

If a file is missing, the app will **not crash** — it shows a friendly
"Model unavailable" warning instead, that model's card is skipped in
detection, and the System Info page marks it 🔴 Missing. Right now you have
7 of the 9 files (no LSTM / Autoencoder yet) — the app works fully with
Random Forest, XGBoost, Isolation Forest, and Random Forest Multiclass, and
will simply mark LSTM and Autoencoder as unavailable until you add those
`.keras` files.

> **Note on your two XGBoost files:** you uploaded both
> `xgboost_binary.pkl` (50 estimators) and `xgboost_binary_model.pkl`
> (200 estimators). The app only loads a file named exactly
> `xgboost_binary.pkl`, so pick whichever one you trust more (e.g. based on
> its validation score) and rename it to `xgboost_binary.pkl` in the
> `models/` folder — don't keep both under different names, since only one
> is ever used.

The 68-feature order (`FEATURE_COLUMNS` in `app.py`) was read directly from
`scaler_binary.feature_names_in_`, so it matches your trained scalers
exactly — including the duplicated `Fwd Header Length.1` column that
CICIDS2017-style CSVs are known to contain. The multiclass model in your
upload was trained on a **web-attack** subset with classes `BENIGN`,
`Web Attack - Brute Force`, `Web Attack - SQL Injection`, and
`Web Attack - XSS`; these are decoded automatically and shown as-is.

### GitHub file-size note

All 7 files you uploaded are well under GitHub's 25 MB drag-and-drop limit
(the largest, `random_forest_multiclass.pkl`, is ~18.5 MB) and under git's
100 MB hard limit, so a normal `git add` / `git commit` / `git push` will
work fine. If you later add larger files (e.g. an LSTM/autoencoder
`.keras` model over ~25 MB), use
[Git LFS](https://git-lfs.com/) for those specific files instead of
regular git.

### Optional files

- `models/metrics.json` — pre-computed evaluation metrics for the
  Model Comparison page, e.g.:

  ```json
  {
    "Random Forest": {"Accuracy": 0.98, "Precision": 0.97, "Recall": 0.96, "F1-Score": 0.965, "ROC-AUC": 0.99},
    "XGBoost": {"Accuracy": 0.98, "Precision": 0.97, "Recall": 0.97, "F1-Score": 0.97, "ROC-AUC": 0.99}
  }
  ```

  If this file is missing, the page displays "Evaluation metrics are not
  available." instead of inventing numbers.

- `models/autoencoder_threshold.pkl` — a single numeric reconstruction-error
  threshold used to turn the Autoencoder's score into a BENIGN/ATTACK
  decision. Without it, the app only shows the raw reconstruction error.

- `data/*.csv` — a training/reference dataset for the Dataset Analysis page
  and for the "Dataset Records" KPI on the Dashboard.

---

## 3. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. How to Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 5. CSV Requirements (Batch Detection & Dataset Analysis)

The uploaded CSV must contain the **68 required feature columns** defined in
`FEATURE_COLUMNS` inside `app.py` (Destination Port, Flow Duration, packet
statistics, TCP flag counts, IAT statistics, etc.). Extra columns are safely
ignored. Missing required columns are reported by name instead of causing a
crash.

For **LSTM**, the CSV must contain at least **10 consecutive records**
(`WINDOW_SIZE = 10`), since the model expects a sequence rather than a
single row.

For **Dataset Analysis**, a `Label` (or `Class`) column is used, if present,
to compute Benign/Attack counts and the class-distribution chart.

---

## 6. Model Selection Guide

| Detection Type            | Available Models                          |
| -------------------------- | ------------------------------------------ |
| Binary Classification      | Random Forest, XGBoost, LSTM               |
| Unsupervised Detection      | Isolation Forest, Autoencoder              |
| Multiclass Classification   | Random Forest Multiclass                   |

The UI only shows models compatible with the selected detection type.

---

## 7. Deployment

The app is a standard Streamlit app and can be deployed to:

- **Streamlit Community Cloud** — push this repo, point it at `app.py`.
- **Docker** — build an image with `requirements.txt` and run
  `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`.
- Any VM/server with Python 3.10+ installed.

Make sure the `models/` folder (and optionally `data/`) is included in the
deployment package — the app loads models from local disk only.

---

## 8. Limitations

- This is an **offline inference tool**. It does **not** perform live
  network monitoring or packet capture.
- Predictions are only as accurate as the models placed in `models/`.
- If `metrics.json` or the autoencoder threshold file are not provided, the
  app deliberately shows "Not available" rather than fabricating numbers.
- The Dataset Analysis page requires a CSV in `data/`; without one, it will
  say the dataset is not available.
- LSTM predictions require a contiguous sequence of at least 10 records and
  are therefore only available in Batch Detection, not single-record
  Traffic Detection.
