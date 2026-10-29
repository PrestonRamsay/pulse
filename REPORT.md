# Pulse Model Delivery Report

## 1. Data

### Months used

**July, August and September 2026**, the three most recent consecutive months published in the
2026 MBTA Bus Arrival/Departure zip (October was not yet available on 2026-10-29). They are loaded
into the Postgres `stop_events` table by `python -m src.load_historical`.

### Row counts before and after cleaning

| Month | Raw rows | Delay outside −30..+120 min (dropped) | Duplicate key (dropped) | Inserted | With `actual` (labelled) |
|---|---:|---:|---:|---:|---:|
| 2026-07 | 2,536,378 | 255 | 625 | 2,535,498 | 2,403,202 |
| 2026-08 | 2,511,333 | 224 | 436 | 2,510,673 | 2,378,287 |
| 2026-09 | 2,485,473 | 212 | 1,812 | 2,483,449 | 2,341,037 |
| **Total** | **7,533,184** | **691** | **2,873** | **7,529,620** | **7,122,526** |

- No row had a missing or unparseable key field.
- 407,094 inserted rows (5.4%) have no `actual`. They stay in the table with `delay_min` and
  `late` NULL, but are never used as training examples: an unrecorded departure is not evidence
  of an on-time one.
- `stop_events` is unique on `(service_date, route_id, direction_id, stop_id, scheduled)` and
  every insert uses `ON CONFLICT DO NOTHING`, so reruns are safe and the first row loaded for a
  key wins. The raw files are *not* unique on that key (see below); 2,873 rows (0.04%) lost.

### What exploring the raw files decided

Exploration ran on all nine 2026 files (22,177,874 rows) before choosing the load window
(`notebooks/01_explore_historical_data.ipynb`).

- **Time encoding.** `scheduled` and `actual` are times of day on a dummy `1900-01-01` date with
  a misleading `Z`; they are Boston wall-clock times, and trips after midnight roll into
  `1900-01-02`. The loader adds the offset from `1900-01-01` to `service_date`, so a 00:45 trip
  on the 15th's service day is stored as the 16th at 00:45 while keeping `service_date` = 15th.
  Timestamps are stored without a UTC offset, matching the files.
- **Missing `actual`.** About 5% of rows in every month; these are kept unlabelled (above).
- **Delay distribution.** Of rows with both times, 0.01% fall outside −30 to +120 minutes and are
  treated as data errors. The retained delays have a median of 2.4 minutes and a 90th percentile
  of 10.3 minutes, so the 3-minute threshold sits close to the median and the classes are
  nearly balanced (about 44% late).
- **Uniqueness.** The key repeats for about 0.3% of rows across the nine months. Most repeats
  differ in `half_trip_id`, `actual` or `standard_type`, so they are not exact duplicates but
  separate records that the files place at the same stop and minute. The task fixes the table
  key, so the first record is kept and the drop is counted rather than widening the key.
- **Half-empty columns.** `half_trip_id` is nearly complete (550 blanks in nine months). `headway`
  (12% blank in September) is missing about equally on `Headway` and `Schedule` rows and most
  often at midpoints; it is the gap to the bus ahead as observed on the day, so it is only as
  complete as the AVL record of the previous trip. It is never a feature. `scheduled_headway`
  (2.6% blank) is missing only on `Schedule`-type rows, where some timepoints are run to the
  clock with no planned headway.
- **Ids.** Route ids are text: single digits are zero-padded (`01`), and the Silver Line and
  crosstown routes are named (`SL1`, `CT3`). Stop ids are numeric strings. Both are kept in file
  form; `src/pull_daily.py` maps API ids back to them.

### Second source: the daily API pull

`python -m src.pull_daily` fetches the day's `/schedules` for every route in the table (148 of
153 file routes map to an API id; `191`–`194`, `746_` and four September-only `80xx` shuttle
routes are unknown to the API and logged), keeps only the (route, direction, stop) timepoints
present in the historical rows, and inserts them with `source = 'api'` and no `actual`. Each run
is appended to `logs/pull_daily.log`.

| Run | Service date | API records fetched | Timepoint rows kept | Rows inserted |
|---|---|---:|---:|---:|
| 1 | 2026-10-29 (Thu) | 384,472 | 92,348 | 92,149 |
| 2 | 2026-10-30 (Fri) | _TODO_ | _TODO_ | _TODO_ |

92,149 rows is in line with a weekday in the files (about 91,400 on 8–10 September). The 199
kept-but-not-inserted rows share a key with another trip at the same stop and minute.

### Splits

Labelled historical rows only (`source = 'historical' AND late IS NOT NULL`), cut on
`service_date` by `python -m src.make_splits`. Features exported: `route_id`, `direction_id`,
`stop_id`, `time_point_id`, `scheduled`, `scheduled_headway`, `point_type`, `standard_type`,
`time_point_order`, `day_of_week`, `hour`, `minute_of_day`; label `late`.

| Split | Service dates | Rows | Late rate | DVC md5 |
|---|---|---:|---:|---|
| train | 2026-07-01 – 2026-08-31 | 4,781,489 | 42.53% | `9c76528a5a338970f93ee5e6e3928ce2` |
| val | 2026-09-01 – 2026-09-15 | 1,155,247 | 48.72% | `e381cfc557b368f1f9360888e12f3d02` |
| test | 2026-09-16 – 2026-09-30 | 1,185,790 | 48.30% | `b58c1d86758c22e0863529294b79a3cc` |

The script asserts the order: train's last `scheduled` is 2026-09-01 02:43 (an after-midnight
trip on 31 August's service day) and validation's first is 2026-09-01 03:10; validation ends
2026-09-16 02:40 and test starts 03:00. The test split is held out until the final evaluation.

The late rate is not stable across the cut: July (42.7%) and August (42.4%) are summer service,
while 1–7 September runs 45.0% and 8–15 September, with schools back, 51.7%. Validation and test
are both September, so the model is trained on summer and judged on the start of term.

## 2. Baselines

Both trained on train, scored on validation, logged to MLflow experiment `pulse-late-3min`
(local `mlflow.db`), exported to `mlflow_runs.csv`. Configs: `configs/baseline_majority.yaml`,
`configs/baseline_logreg.yaml`. Features for the regression: one-hot `route_id`,
`direction_id`, `stop_id`, `point_type`, `standard_type`, `hour`, `day_of_week`
(1,356 columns, `handle_unknown="ignore"`).

| Run | Accuracy | AUC-ROC | PR-AUC |
|---|---:|---:|---:|
| `baseline_majority` | 51.28% | 0.500 | 0.487 |
| `baseline_logreg` (lbfgs, C=1.0, max_iter=300) | 62.58% | 0.681 | 0.652 |

- **Majority class** predicts "on time" for everything (train is 42.5% late), so it scores one
  minus the validation late rate. Because the classes are nearly balanced, 51% is close to a
  coin flip: accuracy comes almost entirely from the model, not from class imbalance.
- **Logistic regression** adds 11.3 points of accuracy and lifts AUC from 0.50 to 0.68. The
  schedule alone (which route, stop, direction, hour and weekday) carries real but modest
  signal: knowing *where and when* a bus runs ranks a late event above an on-time one about
  two times in three.
- **The gap is narrowed by drift, not only by weak features.** The regression learned summer's
  base rate: its mean predicted probability on validation is 0.427 against 48.7% actually late,
  so at a 0.5 cut-off it flags only 35% of events as late. AUC, which ignores the threshold, is
  the fairer measure of signal here; the accuracy gap shows that a model trained on summer
  needs recalibration, or recent late-rate features (spec §3), to track the September jump.
- Unseen categories are not the problem: only 0.16% of validation rows have a route or stop
  absent from train.

## 3. Model Development
- Architectures and configs tried (reference MLflow run names)
- What improved results, what didn't, and your hypothesis why
- Final architecture description + parameter count
- How you handled the high-cardinality features (route, stop), and why

## 4. Final Evaluation (held-out test weeks)
- Accuracy, AUC-ROC, PR-AUC, recall, precision at chosen threshold
- Confusion matrix
- Calibration check: predicted vs. actual late rate by hour
- Route-level sanity check: top-5 predicted-late routes vs. actual

## 5. Threshold Recommendation
- Chosen operating threshold and the rider-experience rationale
- The precision/recall trade at 2-3 alternative thresholds

## 6. Reproducibility
- Git commit, DVC versions, MLflow experiment name, registered model version
- Exact command(s) to reproduce the winning run

## 7. Limitations & Next Steps
- What the model can't do; what you'd try with more time
- How would the picture change if "late" meant five minutes instead of three? Argue it qualitatively