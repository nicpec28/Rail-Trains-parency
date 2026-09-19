# SHM (Structural Health Monitoring) Subsystem — Fatigue Damage Regression

Predicts a continuous cumulative fatigue "damage" value (see `Train_Labels.csv`'s
`damage` column) from a one-column raw stress time series per file, using
rainflow-cycle-counting fatigue features.

## Approach

1. **`feature_extractor.py`** — loads a single-column stress signal and
   extracts time-domain stats plus rainflow-based fatigue features (cycle
   counts binned into 24 fixed, geometrically-spaced stress-amplitude bins,
   so `cycle_bin_X` means the same physical thing across every file).
2. **`model_components.py`** — custom `scikit-learn`-compatible transformers:
   `VIFSelector` (greedy collinearity elimination by variance-inflation
   factor, with protected features/prefixes exempted — e.g. compositional
   histogram-bin fractions that are collinear by construction) and
   `MonotonicAwareRegressor`.
3. **`model_eval.py`** — compares Ridge, ElasticNet, SVR, KNN, Random Forest,
   HistGradientBoostingRegressor, and XGBoost via `RandomizedSearchCV` inside
   a `VIFSelector → scaler → model` pipeline wrapped in a
   `TransformedTargetRegressor`, scored on MAPE with repeated K-fold CV, then
   reports permutation feature importance on the final model.
4. **`predict.py.py`** — loads the saved model bundle and scores every file
   in `Test/`, writing `shm_predictions.csv`.

## Files

- `feature_extractor.py` — rainflow + time-domain feature extraction from a
  raw stress signal.
- `model_components.py` — `VIFSelector`, `MonotonicAwareRegressor`.
- `model_eval.py` — model comparison, hyperparameter search, training.
- `predict.py.py` — inference script (see naming note below).
- `shm_model.pkl` — deployed model bundle used by `predict.py.py`.
- `model_comparison_checkpoint_seed_42.pkl` — a checkpoint from the
  `model_eval.py` comparison run (see naming note below).
- `shm_predictions.csv` — last `predict.py.py` run's output.
- `shm_train_log.txt`, `model_eval_20260919_040930.log` — training logs.
- `Train_Labels.csv` — per-file ground-truth `damage` values for `Train/`.
- `Train/`, `Test/` — raw per-file stress time series.

## Running

```bash
python model_eval.py
python predict.py.py
```

## Folder audit: naming convention, model/pipeline presence

- **`predict.py.py` has a double extension.** Almost certainly an accidental
  rename/save artifact — should be renamed to `predict.py`. As-is, running
  it requires typing the double extension explicitly, and some tooling
  (editors, IDE "run" buttons, packaging) will not treat a `.py.py` file as a
  normal Python module.
- **Inconsistent `shm_` prefixing.** The output artifacts carry an `shm_`
  prefix (`shm_model.pkl`, `shm_predictions.csv`, `shm_train_log.txt`), but
  none of the source scripts do (`feature_extractor.py`,
  `model_components.py`, `model_eval.py`, `predict.py.py`) — the opposite of
  `Door`'s and `Rail Corrugation`'s convention, where the pipeline script
  *itself* carries the subsystem prefix (`door_pipeline.py`,
  `rail_pipeline.py`).
- **Log/artifact mismatch.** `model_eval_20260919_040930.log` ends with
  `Saved model to: best_fatigue_model_seed_42.pkl`, but no file by that name
  exists in this folder — only `model_comparison_checkpoint_seed_42.pkl` and
  `shm_model.pkl` are present. Either the log is stale (from a run whose
  output was later renamed) or the save step's target filename changed after
  this log was written; worth reconciling so the log can be trusted as a
  record of what actually produced the current `shm_model.pkl`.
- **Model file (pkl): present**, but as two separate files with different
  roles — `shm_model.pkl` (the one `predict.py.py` actually loads) and
  `model_comparison_checkpoint_seed_42.pkl` (a comparison-run checkpoint).
  Nothing in either filename states which one is the deployed model, unlike
  `ACV`'s `model/model_meta.json` or `Rail Corrugation`'s
  `cv_results.json`, which each pair a model file with metadata that says so
  explicitly.
- **Pipeline: no consolidated `*_pipeline.py`.** Unlike `Door` and
  `Rail Corrugation`, which each merge feature extraction + training +
  inference into one `*_pipeline.py` file, SHM's equivalent logic is split
  across four files (`feature_extractor.py`, `model_components.py`,
  `model_eval.py`, `predict.py.py`) with no single entry point tying them
  together.
