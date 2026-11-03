# FraudGuard AI

Experimental credit-card fraud detection prototype combining an Autoencoder anomaly signal, a feature-similarity Graph Convolutional Network (GCN), and a Streamlit dashboard backed by FastAPI.

> **Research prototype only.** The hybrid fusion score is not a calibrated fraud probability. The demo dashboard generates synthetic V1–V28 feature values; it does not ingest a real card's transaction history. The INR-to-EUR conversion is a fixed project assumption and does not establish validation for Indian banking transactions.

## Current architecture

- **Dashboard:** Streamlit app in `dashboard/app.py`.
- **API:** FastAPI app in `api/main.py`.
- **Inference:** `engine/hybrid_inference.py` loads the Autoencoder, GNN checkpoint, scaler, and sampled training reference graph.
- **Data:** Chronological train, validation, and test partitions under `data/processed/`.
- **Artifacts:** Model weights and scaler under `models/weights/`; evaluation outputs under `results/`.

## Local setup (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

Run the API in terminal 1:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.

Run the dashboard in terminal 2:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run dashboard/app.py
```

Streamlit normally opens at `http://localhost:8501`.

## Model and evaluation notes

- The data split is chronological: approximately 64% train, 16% validation, 20% test, with a 0.1% row-count embargo at partition boundaries.
- The Autoencoder is trained using legitimate transactions only.
- The GNN uses a k-nearest-neighbor feature-similarity graph. Edges indicate feature similarity, not verified account/card relationships.
- GNN scaling is fit on training data. Hybrid inference attaches each query to a fixed training reference graph.
- The final hybrid decision threshold is 0.5. The GNN standalone threshold is 0.9378564358.
- The untouched chronological test partition contains 56,678 rows and 75 fraud cases. See [the frozen test report](results/frozen_hybrid_test_results.json).
- The hybrid test result at threshold 0.5: precision 0.6250, recall 0.7333, F1 0.6748, ROC-AUC 0.9812, average precision 0.5997, with 55 true positives and 33 false positives.
- GNN achieved higher recall (0.7467) and average precision (0.7427); hybrid achieved higher precision and F1. These metrics are based on one holdout and should be interpreted cautiously.

## Known limitations and next work

1. The demo currently supplies zero-valued V1–V28 features by default, or a small hand-crafted unusual-feature toggle. It is not a real transaction feature pipeline.
2. The hybrid API currently emits `APPROVE` or `OTP_VERIFY`; the automatic `BLOCK` policy in `core/config.py` is not wired into this endpoint.
3. Transaction history is session-scoped in Streamlit; it is not durable storage.
4. API URL and currency conversion assumptions should become environment-configurable.
5. Add genuine automated unit/integration tests; the current `tests/` directory contains evaluation/training scripts rather than a conventional pytest suite.
6. Add deployment configuration, CI, and operational monitoring only after the inference/API contract is stabilized.

## Repository layout

- `api/`: FastAPI routes and request validation
- `dashboard/`: Streamlit UI
- `core/`: project configuration
- `data/`: preprocessing and partitions
- `engine/`: inference and temporal feature store
- `models/`: model training and saved artifacts
- `results/`: evaluation reports
- `scripts/`: training, scoring, evaluation and validation utilities
- `tests/`: legacy evaluation/training scripts; automated test coverage remains to be added
