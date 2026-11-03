# Project status and research handoff

## Work already present on GitHub main

- Chronological data preparation and partition verification scripts.
- Autoencoder, GNN, supervised-model and fusion artifacts/scripts.
- FastAPI endpoints and the Streamlit transaction simulator.
- Multiple historical validation/test reports.
- Stage 1 unified evaluation script and the earlier research audit artifacts.

## Completed work that was not yet represented clearly in the repository

The following results were completed in the working project but were missing as canonical, clearly named records on `main` at the time of this audit:

1. Frozen three-way Autoencoder/GNN/Hybrid test comparison at the fixed validation-selected thresholds.
2. Deployment-style inductive GNN validation against the sampled training reference graph.
3. Numerical equivalence check for optimized local GCN query inference.
4. Explicit caveats about synthetic dashboard features, uncalibrated hybrid score, and the unimplemented BLOCK action.

Canonical machine-readable records are in:
- `results/frozen_hybrid_test_results.json`
- `results/deployment_gnn_validation.json`
- `results/local_gnn_equivalence.json`

## Frozen test result summary

Test partition: 56,678 rows; 75 fraud cases; 56,603 legitimate cases. Thresholds were frozen before test evaluation and model artifact hashes were unchanged.

| Model | Precision | Recall | F1 | ROC-AUC | Average precision | TP | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | 0.5106 | 0.3200 | 0.3934 | 0.9333 | 0.2634 | 24 | 23 |
| GNN | 0.5833 | 0.7467 | 0.6550 | 0.9822 | 0.7427 | 56 | 40 |
| Hybrid | 0.6250 | 0.7333 | 0.6748 | 0.9812 | 0.5997 | 55 | 33 |

Interpretation: hybrid is the best balanced operating point by precision/F1; GNN has the strongest recall and ranking metrics. The test set has only 75 fraud cases, so metrics are uncertain and one holdout does not establish future performance.

## Validation evidence

- Deployment-style GNN validation: 45,285 validation rows, 52 fraud; precision 0.5070, recall 0.6923, F1 0.5854, AP 0.6718, FPR 0.0774%.
- Optimized local GCN equivalence: 256/256 sampled queries within atol 1e-5 and rtol 1e-4; max absolute probability difference 5.96e-8. This is an implementation-equivalence check, not a predictive-performance test.

## Repository gaps identified

- No top-level README was present.
- `requirements.txt` omitted `altair`, although `dashboard/app.py` imports it directly.
- Existing `tests/` entries are mostly evaluation/training scripts; no conventional automated pytest suite was identified.
- No `.github/workflows/` CI workflow was present.
- No `docs/` project handoff or architecture note was present.
- Existing `results/live_style_final_test_results.json` reports a different 0.55 threshold and must not be confused with the frozen 0.50 hybrid test evaluation documented here.
- `core/config.py` points to `gnn_model.pt`, while the current hybrid engine loads `gnn_model_state.pt`; confirm whether the former is used by the legacy inference path before changing it.
- The dashboard currently creates synthetic V1–V28 demo values, uses a fixed local API URL, and stores session history in Streamlit session state.
- The hybrid endpoint does not implement automatic BLOCK despite the general thresholds in configuration.

## Safety of future changes

Do not retrain models or tune thresholds against the frozen test partition. Any model/threshold changes should be evaluated on a later chronological holdout or a properly nested validation design. Do not claim production readiness or Indian banking validation from the current public-dataset experiment.
