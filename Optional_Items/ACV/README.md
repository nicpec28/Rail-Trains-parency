# ACV Refrigerant-Leak Localisation

Ranks the 8 cars of a train from most- to least-likely to have a refrigerant
leakage fault, for the hackathon's ACV subsystem track (see
`ACV_Subsystem_Info_Kit.md`).

## Approach

Refrigerant leakage causes undercharge, which shows up physically as reduced
cooling capacity: the affected car struggles to reach its own cooling
setpoint relative to its peers on the same train, at the same time, under the
same ambient conditions — and, when richer telemetry is available, as
short-cycling of the compressor and depressed low-side refrigerant pressure.

With only 6 labeled fault cases available, model choice is driven by a
build order, not a single architecture picked up front:

1. **Engineer schema-agnostic, physically-motivated features per car**
   (`acv_fault/features.py`), then z-score each one *within its own file*
   (across that file's 8 cars) so files with very different ambient regimes,
   sensor scales, and even different parameter sets are comparable. Column
   headers are always read dynamically (`acv_fault/data_loading.py`) rather
   than assumed.
2. **Score several unsupervised, per-file anomaly-detection methods first**
   (`acv_fault/baselines.py`) — these need no labels to fit at all, so there
   is zero overfitting risk and they work immediately on any file's schema.
3. **Only keep a supervised combination layer on top if it measurably beats
   the best unsupervised baseline**, checked with leave-one-file-out
   cross-validation scored on the competition's own rank-decay metric — not
   accuracy, since that's what the submission is actually graded on.

### Features (`acv_fault/features.py`), per car, per file

| Feature | Rationale |
|---|---|
| `cool_gap_mean` / `cool_gap_p90` / `cool_gap_max` | Indoor/cabin temp minus target cooling temp, while actively cooling. The core leak signature: worse cooling than peers. |
| `fault_rate` | Average active-rate across any column whose name contains "fault" (compressor/fan/heater fault flags, when present). |
| `mode_invalid_rate` | Rate of the car's running mode reporting "invalid". |
| `info_invalid_rate` | Rate of the car's "information valid" status reporting "invalid". |
| `load_active_rate` | Rate of "load halved" / "load shedding" being active — a controller response to a struggling unit. |
| `missing_data_rate` | Fraction of entirely-null rows for that car (diagnostic; some training files have cars with no telemetry at all). |

This is the complete published parameter set from the info kit's "standard"
schema (Section 2.1), transformed the most direct way (gap, invalid-rate,
active-rate) — not a hand-picked subset. Features that don't exist in a given
file's schema are neutral (z-score 0) rather than causing an error.

**Removed after a leakage audit:** an earlier version also included
`cool_gap_persistence`, `cooling_duty_cycle`, `cabin_temp_std`,
`low_pressure_mean`, `low_pressure_p10`, `high_pressure_mean`, and
`compressor_cycle_rate`. The last four were built by inspecting the one
training file with richer telemetry (`acv_case_04.xlsx`) and finding what
would rank its *known* faulty car correctly — with only one file in the
whole dataset carrying that data, there's no independent file left to test
whether those features generalize versus just fit that one example, so
keeping them would mean reporting a validation score that partly validates
itself. A controlled ablation (dropping all 7 and re-scoring) showed this
costs **nothing**: Mahalanobis distance scores identically with or without
them. They were removed rather than kept on an unverifiable assumption.

### Analytical models compared (`acv_fault/train.py`)

Every method below is evaluated on the same 6 labeled cases with the
competition's rank-decay metric, so the comparison is apples-to-apples:

| Method | Type | Score |
|---|---|---|
| Single feature (`cool_gap_mean` only) | heuristic | 0.896 |
| Aggregate absolute z-score | unsupervised | 0.917 |
| **Mahalanobis distance** (Ledoit-Wolf shrinkage covariance) | unsupervised | **0.979** |
| Isolation Forest | unsupervised | 0.958 |
| Local Outlier Factor | unsupervised | 0.979 |
| PCA reconstruction error | unsupervised | 0.458 |
| Logistic regression (L2, `class_weight="balanced"`), nested CV | supervised | 0.979 |

The unsupervised methods score every file's 8 cars purely against each
other — no fitting across files, no labels used at all, so there's nothing
to leave out or overfit. Logistic regression's score is from **nested**
leave-one-file-out CV: for each held-out file, its regularization strength
`C` is chosen using only the *other 5* files, never the held-out file's own
score — a plain (non-nested) LOFO would pick `C` by looking at all 6 outer
scores at once and then report that same number back, which is a mildly
optimistic estimate of its own accuracy. Logistic regression now ties
Mahalanobis exactly (0.979 vs 0.979); on a tie, `train.py` prefers the
unsupervised method, since it needs no training data, can't go stale, and
adapts to a new file's schema with zero retraining.

`train.py` runs this comparison automatically and picks whichever wins:
right now that's **Mahalanobis distance**, which is what `model/acv_model.pkl`
actually contains. Mahalanobis distance also captures *correlated* deviations
across features better than treating each feature independently, which plain
z-scoring cannot do.

Run the full comparison yourself with:

```bash
python -m acv_fault.train --data-dir . --labels Train_Labels.csv --model-out model/acv_model.pkl
```

### Known limitation

One of the six training cases (`acv_case_04.xlsx`) uses a much richer
~60-parameter-per-car schema, and 4 of its 8 cars report no telemetry at all
for that case. Its faulty car does not show an elevated `cool_gap` the way
every other training case does — under the current (leakage-audited) feature
set it lands in 2nd place rather than 1st, which is the honest, unresolved
cost of removing the pressure/compressor features that used to mask this
(see "Removed after a leakage audit" above). This is disclosed rather than
patched, since patching it would mean reintroducing the same leakage.

### Validation caveats (read before trusting the 0.979 headline number)

- **Sample size.** 6 labeled files is not enough to distinguish "genuinely
  robust" from "got lucky on 5 easy files" — 5 of the 6 have an obvious
  `cool_gap` signal that almost any reasonable method picks up; only 1 file
  (`acv_case_04.xlsx`) is a real stress test, and it's a miss (2nd place).
  Treat 0.979 as an optimistic training-set comparison, not a precise
  estimate of held-out test performance.
- **No claim of pipeline-level leakage.** A full audit (labels into features,
  cross-file information into a per-file score, hyperparameters tuned by
  peeking at the score they're then reported on) found the deployed scoring
  path itself clean: Mahalanobis distance is recomputed from scratch on
  whichever file it's given, with zero contact with the 6 training files or
  their labels at inference time. The two issues that did exist (feature
  selection informed by a known label; non-nested hyperparameter selection)
  were both in the *evaluation/design* process, not the deployed model's
  mechanics, and both are fixed as described above.

## Files

- `acv_fault/data_loading.py` — reads a case `.xlsx`'s own headers and splits
  it into a per-car parameter frame.
- `acv_fault/features.py` — per-car feature engineering + within-file
  z-scoring.
- `acv_fault/baselines.py` — unsupervised per-file anomaly-scoring methods
  (Mahalanobis, Isolation Forest, LOF, PCA reconstruction, aggregate z-score).
- `acv_fault/scoring.py` — the competition's rank-decay scoring formula.
- `acv_fault/train.py` — builds the training table, compares unsupervised
  baselines against a supervised combination layer via leave-one-file-out
  CV, saves whichever wins.
- `predict.py` — inference CLI (see below).
- `model/acv_model.pkl`, `model/model_meta.json` — the chosen model and its
  metadata (full comparison table, CV scores, chosen method).

## Running inference

```bash
python predict.py --input acv_test_case.xlsx --output acv_predictions.csv
```

`--input` also accepts a directory of `.xlsx` files, in which case one row is
written per file. Output format matches the info kit's Section 3 exactly:
`file_id,ranked_cars`, with car identifiers pipe-separated and ranked
most-to-least likely, exactly as they appear in that file's own column
headers.

**Note:** the top-level hackathon README (with the exact `predict.py`
`--input`/`--output` contract and full deliverables list) wasn't available in
this workspace — the CLI above is a reasonable default and should be checked
against that spec before submission.

## Folder audit: naming convention, model/pipeline presence

- **File naming.** The `acv_fault/` package (`baselines.py`, `data_loading.py`,
  `extra_metrics.py`, `features.py`, `scoring.py`, `train.py`) is consistently
  `snake_case` with no redundant `acv_` prefix (the package name itself
  provides that). The top-level `predict.py` breaks from the `door_pipeline.py`
  / `rail_pipeline.py` naming style used in the sibling `Door` and
  `Rail Corrugation` folders — there is no single `acv_pipeline.py`; training
  and inference are two separate entry points (`acv_fault/train.py` and
  `predict.py`) instead of one consolidated file.
- **Duplicate package name.** `acv_rectified/` contains its own full copy of
  the `acv_fault` package (`acv_rectified/acv_fault/*.py`), distinct from the
  top-level `acv_fault/`. Two directories both defining a module named
  `acv_fault` risks import ambiguity if both ever end up on the same
  `PYTHONPATH` at once, and nothing in the naming states what "rectified"
  actually changed relative to the original — worth a one-line note in
  `acv_rectified/model_meta.json` or a comment at the top of its `train.py`.
- **Archives.** `acv_fault.zip` and `acv_rectified.zip` appear to be zipped
  snapshots of the two unpacked directories above — duplicated content in two
  forms. Fine as delivery artifacts, but worth confirming they're kept in
  sync with the unpacked folders (or removed once no longer needed).
- **Model file (pkl): present.** `model/acv_model.pkl` (deployed) and
  `acv_rectified/acv_model_rectified.pkl` (rectified variant), each paired
  with its own `model_meta.json`.
- **Pipeline: present, split across files rather than one script.** No file
  is literally named `*_pipeline.py`; the equivalent logic lives in
  `acv_fault/train.py` (build/compare/select) + `predict.py` (inference).
  Functionally complete, just named differently from `Door`'s and
  `Rail Corrugation`'s single-file `*_pipeline.py` convention.
