# Door Subsystem — Abnormal Resistance Detection

Classifies each door open/close cycle as `Normal` or `Abnormal resistance`
from a continuous stream of door-sensor telemetry (motor current/voltage,
back-EMF, leaf position, open/close commands, and status flags).

## Approach

`door_pipeline.py` runs the whole pipeline, in one file, via subcommands:

1. **Segment** the continuous `Train.csv`/`Test.csv` stream into individual
   cycles by detecting timestamp gaps larger than 2s (verified 110/110 exact
   match against `Train_Segments_Answer.csv`).
2. **Extract ~121 candidate features per cycle** (current/voltage/back-EMF
   time-series stats, an energy proxy, door-position displacement/overshoot/
   reversals, command fractions, and flag-transition counts).
3. **Drop near-constant / highly-collinear features** (label-free, computed
   on the full dataset — not a leakage risk).
4. **Fit inside every CV fold**: scale → SMOTE-balance the *training* fold
   only → `SelectKBest` → L1-penalized logistic regression
   (`class_weight="balanced"`). The held-out fold is never resampled.
5. **Refit on all 110 labelled cycles** for the deployed model, since there's
   no natural grouping (file/operating-condition) to hold out a validation
   split by, and 110 cycles (30 Abnormal) is too few to spend on a single
   fixed split — see the design-decision note at the top of
   `door_pipeline.py` for the full justification.

Validation is a distribution, not a point estimate: 5-fold × 10-repeat
stratified CV (50 train/validation splits), reported as mean ± std
(`cv_results.json`: F1[Abnormal] = 0.998 ± 0.013 across those 50 splits).

## Files

- `door_pipeline.py` — segmentation + feature engineering + training (CV,
  final fit) + inference, run via `train` / `predict` subcommands.
- `door_model.pkl` — the fitted pipeline (scaler + SMOTE + feature selector +
  logistic regression) bundled with its feature-column order, via the
  `DoorModel` wrapper class.
- `cv_results.json` — cross-validation summary, kept/dropped feature lists.
- `door_predictions.csv` — last `predict` run's output.
- `Train.csv` / `Test.csv` — raw continuous sensor streams.
- `Train_Segments_Answer.csv` — per-cycle ground-truth labels for `Train.csv`.

## Running

```bash
python door_pipeline.py train --train-csv Train.csv --answers Train_Segments_Answer.csv --model-out door_model.pkl --results-out cv_results.json
python door_pipeline.py predict --input Test.csv --output door_predictions.csv --model door_model.pkl
```

## Folder audit: naming convention, model/pipeline presence

- **File naming.** Outputs consistently carry a `door_` prefix
  (`door_pipeline.py`, `door_model.pkl`, `door_predictions.csv`). The one
  exception is `cv_results.json`, which has no `door_` prefix — harmless
  since the folder only has one such file, but inconsistent with the rest of
  the folder's naming, and with `Rail Corrugation`'s equally unprefixed
  `cv_results.json` (same pattern, different folder). The raw data files
  (`Test.csv`, `Train.csv`, `Train_Segments_Answer.csv`) use `PascalCase` —
  reasonable since they look like externally-supplied inputs rather than
  pipeline outputs, but it's a different casing convention from the
  `snake_case` used everywhere else in this folder.
- **Model file (pkl): present.** `door_model.pkl`.
- **Pipeline: present.** `door_pipeline.py` — a single consolidated
  train+predict script (explicitly merged from three original scripts to
  avoid sibling-module import issues; see its module docstring).
