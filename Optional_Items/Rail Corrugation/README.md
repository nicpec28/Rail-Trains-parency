# Rail Corrugation Subsystem — Side I / Side II / Normal Classification

Classifies each 1-second, 10,000 Hz axle-box vibration+shock recording (64
positions = 8 cars × 8 positions/car, rail Side I = odd positions, Side II =
even positions per car) as `Normal`, `Side I`, or `Side II` corrugation.

## Approach

`rail_pipeline.py` runs the whole pipeline, in one file, via subcommands
(`build-features`, `train`, `predict`):

1. **Parse car/position directly out of each column's header text** (regex
   on strings like `"Shock of bearing in position 3 of car 5"`), so a
   reordered or partial column set can't silently misassign a channel.
2. **Extract 9 time/frequency-domain stats per channel** (mean, std, RMS,
   peak-to-peak, crest factor, kurtosis, dominant FFT frequency, spectral
   energy, spectral centroid) across all 128 vibration/shock channels plus
   the raw rotating-speed signal → 1,161 raw features per file.
3. **Narrow to ~50 features via `SelectKBest`** (ANOVA F-test), refit inside
   each CV fold so the validation fold is never seen by the selector.
4. **Handle the class imbalance by weighting, not resampling** (234 Normal /
   24 Side II / 14 Side I out of 272 files; Side I is only 5.1%):
   `class_weight="balanced"` (or manually-computed `sample_weight` for
   XGBoost), `StratifiedKFold`, loosened tree leaf/split minimums, and macro
   F1 as the model-selection metric. See the header of `rail_pipeline.py`
   for the full rationale.
5. **Model comparison across 3 seeds** (lasso, ridge, decision tree, random
   forest, SVM, XGBoost) — SVM wins (mean macro F1 = 0.621 ± 0.050;
   `C=1, gamma=0.01`), and that's what's fit on all 272 files and saved to
   `rail_model.pkl` (see `rail_train_log.txt` / `cv_results.json`).

`rail_enhanced/` is a second iteration with an enlarged feature set (1,601
raw features vs. 1,161) plus VIF-based collinearity filtering; SVM wins there
too, with a similar score, saved separately as `rail_model_enhanced.pkl` so
the original run isn't overwritten.

## Files

- `rail_pipeline.py` — feature building + training (CV, model selection,
  final fit) + inference, run via `build-features` / `train` / `predict`
  subcommands.
- `rail_model.pkl` — the fitted SVM pipeline.
- `cv_results.json`, `rail_train_log.txt` — cross-validation summary and full
  training log (per-seed, per-model scores; selected model + hyperparameters).
- `train_features_full.csv` — the 1,161-feature table built from `Train/`
  (output of `build-features`).
- `Train_Labels.csv` — per-file ground-truth labels for `Train/`.
- `Train/`, `Test/` — raw per-file sensor recordings (272 / 68 files).
- `rail_enhanced/` — second iteration with an enlarged feature set:
  `rail_pipeline_enhanced.py`, `rail_model_enhanced.pkl`,
  `cv_results_enhanced.json`.
- `rail_enhanced.zip` — zipped snapshot of `rail_enhanced/`.
- `__pycache__/` — compiled bytecode cache, not source; safe to delete.

## Running

```bash
python rail_pipeline.py build-features --start 1 --end 272 --out train_features_full.csv
python rail_pipeline.py train
python rail_pipeline.py predict --input Test/ --output rail_predictions.csv --model rail_model.pkl
```

## Folder audit: naming convention, model/pipeline presence

- **Folder name.** `Rail Corrugation` contains a space, unlike the sibling
  `ACV`, `Door`, and `SHM` folders — worth renaming to `Rail_Corrugation` or
  `RailCorrugation` if these four are ever treated as importable paths
  (a bare space is legal on the filesystem but awkward in shell commands and
  in `import`/`sys.path` usage).
- **File naming.** `rail_` prefix is used consistently for the pipeline,
  model, and log (`rail_pipeline.py`, `rail_model.pkl`,
  `rail_train_log.txt`), and the `rail_enhanced/` variant consistently
  suffixes `_enhanced` on every one of its equivalents. The one unprefixed
  file is `cv_results.json` (same pattern as `Door/cv_results.json`).
  `train_features_full.csv` (lowercase) and `Train_Labels.csv` (PascalCase)
  sit side by side with inconsistent casing for what are both CSV data
  tables.
- **Build artifact left in place.** `__pycache__/rail_pipeline.cpython-313.pyc`
  is a compiled-bytecode cache, not source — normally excluded from version
  control (`.gitignore`'d) rather than committed alongside the pipeline.
- **Model file (pkl): present**, both for the base run (`rail_model.pkl`) and
  the enhanced run (`rail_enhanced/rail_model_enhanced.pkl`).
- **Pipeline: present**, both for the base run (`rail_pipeline.py`) and the
  enhanced run (`rail_enhanced/rail_pipeline_enhanced.py`) — each a single
  consolidated build-features/train/predict script.
