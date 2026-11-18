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
  Checked on 2026-10-30: reloading September inserted 0 of its 2,485,473 rows, and rerunning
  that day's API pull inserted 0 of 92,484.

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
| 2 | 2026-10-30 (Fri) | 384,859 | 92,484 | 92,266 |

Both runs are in line with a weekday in the files (about 91,400 on 8–10 September). The 199 and
218 kept-but-not-inserted rows share a key with another trip at the same stop and minute.

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

All runs are in MLflow experiment `pulse-late-3min`, trained on train (1 July – 31 August;
runs with history features drop the first 14 days, whose windows are nearly empty) and scored
on validation (1–15 September). The test split was not read until §4. Configs are in
`configs/<run name>.yaml`; the run table is exported to `mlflow_runs.csv`.

### Runs

| Run | Architecture and features | AUC-ROC | Accuracy (0.5) | PR-AUC | Brier |
|---|---|---:|---:|---:|---:|
| `baseline_majority` | constant "on time" | 0.500 | 51.28% | 0.487 | — |
| `history_network_8wk` | one network late rate, last 56 days; no training | 0.518 | 51.28% | 0.499 | 0.252 |
| `baseline_logreg` | logistic regression, one-hot schedule | 0.681 | 62.58% | 0.652 | — |
| `logreg_schedule` | + day type, scheduled headway, timepoint order | 0.683 | 62.92% | 0.654 | 0.228 |
| `gbt_schedule` | gradient-boosted trees, schedule only | 0.706 | 64.65% | 0.683 | 0.221 |
| `history_slot_8wk` | slot's late rate over the last 56 days; no training | 0.723 | 65.85% | 0.703 | 0.216 |
| `logreg_history` | logistic regression + late rates + history counts | 0.725 | 65.07% | 0.705 | 0.221 |
| `gbt_history` | trees + late rates (31 leaves, 300 rounds, lr 0.1) | 0.725 | 66.00% | 0.704 | 0.216 |
| `gbt_history_deep` | trees + late rates (127 leaves, 600 rounds, lr 0.05, L2 1.0) | 0.725 | 66.09% | 0.704 | 0.217 |
| **`logreg_history_rates`** | **logistic regression + late rates** | **0.726** | **66.22%** | **0.706** | **0.214** |

Brier was not logged for the two Task 2 runs. "Late rates" are the spec §3 history features
built in `src/features.py`: network, route, stop and slot (route, direction, stop, hour, day
type) late rates over the 7, 14, 28 and 56 service days before each trip's own day, each
shrunk toward the next-coarser level (pseudo-count 20).

### What helped, what didn't, and why

- **Recent history helped most.** It lifted the logistic regression from 0.683 to 0.726 AUC.
  The slot's 8-week rate alone, with no model, already reaches 0.723. In permutation
  importance on `gbt_history`, shuffling `rate_slot_56d` costs 0.167 AUC; the next feature
  (route) costs 0.008. Once a slot's recent record is known, the schedule adds little.
- **Architecture helped only without history.** Trees beat logistic regression by 0.024 AUC
  on schedule fields alone, because they combine route, hour and timepoint position
  without hand-built interactions. With history the gap disappears (0.725 vs 0.726), and
  bigger trees changed nothing, so the limit is the information in the features, not model
  capacity. What's left is day-to-day variation (traffic, weather, incidents) that nothing
  known the day before predicts.
- **Extra schedule fields barely helped:** day type, headway and timepoint order added 0.002 AUC.
- **History counts hurt.** `logreg_history` also fed the number of trips behind each rate.
  Those counts grew as the windows filled from 1 July, so every September row looked
  "later" than any training row, and the model over-predicted (mean 0.57 against 0.49
  actual). Dropping them (`logreg_history_rates`) fixed it.
- **The school calendar was not used.** Every training day is summer, so a school flag would
  be constant in train and nothing could be learned from it. The recent late rates carry the
  September change instead.
- **Drift and calibration.** All models trained on summer (42.5% late) under-predict
  September. History features close part of the gap (mean prediction 0.449 against 0.487 on
  validation). A one-off calibration fitted on 1–7 September did not carry into the school
  week (calibration error on 8–15 September 0.074 → 0.071), because the jump came after the
  fitting week. **Rolling recalibration** (`src/calibrate.py`) shifts each day's predictions
  on the log-odds scale by the amount that would have made the previous 7 days' predictions
  match their actual late rate, using only earlier days. On 8–15 September it cut
  calibration error to 0.052 for the logistic regression and 0.061 for `gbt_history`.

### Final architecture

`logreg_history_rates`, registered as `pulse-latenet`:

- **Inputs:** one-hot route, direction, stop, point type, standard type, hour, day of week
  and day type (1,338 columns, `handle_unknown="ignore"`), plus 18 standardized numeric
  features: scheduled headway, timepoint order, and the 16 late rates on the log-odds scale.
- **Model:** L2-regularized logistic regression (lbfgs, C = 1.0, max_iter 300).
  **1,357 parameters** (1,356 weights + 1 intercept).
- **Serving step:** rolling 7-day recalibration (one log-odds shift per day, computed from
  the previous week's predictions and outcomes), then a 0.40 warning threshold (§5).

It ties the trees on AUC, is better calibrated before and after recalibration, and each
weight can be read directly. That's why it was chosen over `gbt_history`.

### High-cardinality features (route, stop)

There are 155 routes and 1,163 stop ids (2,483 route/direction/stop combinations).

- **Logistic regression:** one-hot. With 4.8M training rows, even small stops have hundreds
  of examples, and L2 regularization keeps rare columns near zero. Unknown stops at serving
  time score as all zeros (0.16% of validation rows).
- **Trees:** route as a native categorical (under the 255-category limit). Stop enters only
  through its late rates.
- **Both:** the stop and slot late rates work as a **time-aware target encoding**. They
  summarize thousands of categories in a few numbers, use only earlier days (so no target
  leakage), shrink thin categories toward their parent level, and update as the network
  changes, which a fixed one-hot weight cannot.

## 4. Final Evaluation (held-out test weeks)

Run once by `python -m src.evaluate logreg_history_rates --threshold 0.40 --register` on the
test split (16–30 September, 1,185,790 rows, 48.30% late). Model and threshold were fixed
on validation beforehand. Features for test rows use only earlier service days. The rolling
recalibration for a test day uses the previous 7 days' outcomes, which from 23 September on
are earlier test days. That's what production would see, and a row's own label is never used.

### Metrics

| Model | Threshold | Accuracy | AUC-ROC | PR-AUC | Precision | Recall | Warned | Brier | Calib. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`pulse-latenet` (recalibrated)** | 0.40 | 66.17% | **0.738** | **0.715** | 61.84% | **78.23%** | 61.1% | **0.208** | **0.023** |
| `pulse-latenet`, no recalibration | 0.40 | 67.29% | 0.738 | 0.716 | 65.50% | 68.18% | 50.3% | 0.212 | 0.064 |
| `history_slot_8wk` | 0.5 | 66.55% | 0.732 | 0.707 | 68.03% | 58.02% | 41.2% | 0.210 | 0.031 |
| `baseline_logreg` | 0.5 | 61.77% | 0.668 | 0.631 | 64.29% | 46.91% | 35.2% | 0.233 | 0.060 |
| `baseline_majority` | 0.5 | 51.70% | 0.500 | 0.483 | — | 0% | 0% | — | — |

**AUC margin, 95% interval from resampling whole test days (200 draws):**

- over `baseline_logreg`: **+0.069** (0.064 to 0.076)
- over `history_slot_8wk`: **+0.006** (0.0003 to 0.010)

The model beats every baseline on test, and the margins hold on data it never saw. Test AUC
is actually higher than on validation (0.738 vs 0.726). Against the slot's 8-week rate the
margin is real but small. Most of what the model knows is that slot's recent record.

Accuracy at 0.40 is slightly below the 8-week baseline's at 0.5 by design: the threshold
trades accuracy for catching late trips (78% vs 58% recall). At 0.5 the recalibrated model
scores 67.3%. Recalibration brought the mean prediction from 0.419 to 0.481 against 0.483
actual.

### Confusion matrix (threshold 0.40)

| | Predicted on time | Predicted late (warned) |
|---|---:|---:|
| **Actually on time** | 336,585 (28.4%) | 276,436 (23.3%) |
| **Actually late** | 124,711 (10.5%) | 448,058 (37.8%) |

### Calibration by hour (recalibrated, test)

| Hour | Rows | Actual | Predicted | | Hour | Rows | Actual | Predicted |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 36,785 | 34.7% | 37.5% | | 12 | 53,958 | 44.8% | 51.0% |
| 1 | 14,580 | 35.0% | 38.4% | | 13 | 55,870 | 47.0% | 53.6% |
| 2 | 1,610 | 28.0% | 31.0% | | 14 | 61,819 | 50.7% | 53.9% |
| 3 | 357 | 42.9% | 61.5% | | 15 | 66,058 | 53.6% | 53.1% |
| 4 | 5,487 | 34.4% | 38.3% | | 16 | 68,497 | 51.8% | 51.7% |
| 5 | 43,102 | 32.1% | 32.0% | | 17 | 69,229 | 55.1% | 54.9% |
| 6 | 72,559 | 41.8% | 37.2% | | 18 | 70,740 | 54.8% | 54.2% |
| 7 | 75,251 | **51.2%** | **39.9%** | | 19 | 61,132 | 55.9% | 54.6% |
| 8 | 73,598 | **55.1%** | **43.4%** | | 20 | 50,874 | 49.5% | 56.0% |
| 9 | 64,751 | 51.2% | 46.8% | | 21 | 45,416 | 45.4% | 49.8% |
| 10 | 56,583 | 46.3% | 48.8% | | 22 | 43,012 | 44.8% | 48.6% |
| 11 | 54,591 | 44.7% | 49.0% | | 23 | 39,931 | 41.9% | 45.0% |

The afternoon peak (15–19) is well calibrated. The **morning peak is under-predicted by
11–12 points** (7–8am), and midday (11–13) is over-predicted by 4–7 points. The likely cause
is school: summer had no school morning rush, the 8-week windows on 16 September are still
about three-quarters summer, and the daily recalibration moves every hour by the same amount.

By day, weekdays are close (e.g. 16–19 September: 52–53% actual, 51–52% predicted), but
**weekends are over-predicted**: Saturday 26 and Sunday 27 September ran 31% late against
49% and 43% predicted. The recalibration shift is network-wide and weekday-dominated.

### Route-level sanity check (routes with ≥ 2,000 test rows)

| Top 5 by predicted late rate | Predicted | Actual | | Top 5 by actual late rate | Actual | Predicted |
|---|---:|---:|---|---|---:|---:|
| 37 | 72.3% | 73.4% | | 29 | 74.7% | 66.6% |
| 31 | 70.2% | 67.7% | | 37 | 73.4% | 72.3% |
| 35 | 69.6% | 62.4% | | 47 | 70.3% | 63.0% |
| 36 | 67.4% | 66.2% | | 85 | 70.0% | 62.1% |
| 44 | 66.8% | 60.9% | | 39 | 68.5% | 62.7% |

One route (37) is on both lists, and every route the model ranks worst really is bad (61–73%
late against a 48% network average). The routes it misses from the actual top 5 are all
predicted at 62–67%, so they would still get warnings; it ranks them just below the top.
Its predicted spread is compressed: the worst routes run a few points later than predicted.

## 5. Threshold Recommendation

**Recommended threshold: 0.40** on the recalibrated probability, chosen on validation
before the test run. At or above it, a trip shows a lateness warning; below it, "Usually on
time".

**Rationale.** The spec's expensive mistake is telling a rider a trip is reliable and then
having it run late, so the threshold leans pessimistic: a missed late trip costs more than an
unneeded warning. At 0.40 the model catches about 78% of late trips. The cost is many
warnings: about 61% of trips. That is inherent, not a model flaw: about half of all trips
really are late, so an honest model has to warn often.

| Threshold | Validation: warned | Recall | Precision | "On time" but late | Test: warned | Recall | Precision | "On time" but late |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 73.0% | 86.5% | 57.7% | 24.3% | 74.5% | 88.6% | 57.4% | 21.7% |
| **0.40** | **59.5%** | **75.7%** | **61.9%** | **29.3%** | **61.1%** | **78.2%** | **61.8%** | **27.0%** |
| 0.50 | 46.0% | 62.7% | 66.4% | 33.6% | 47.2% | 65.0% | 66.5% | 32.0% |
| 0.60 | — | — | — | — | 33.5% | 49.6% | 71.5% | 36.6% |

"On time" but late = share of trips shown as "Usually on time" that ran late, the spec's main
rider-facing number. Validation figures are for all of 1–15 September after recalibration;
0.60 was only tabulated on 8–15 September (72.8% precision). Test figures are shown for
information only and were not used to choose.

- **0.30** protects riders most (only 22–24% of "on time" trips late) but warns on three
  trips in four, which risks warnings being ignored.
- **0.50** warns on fewer than half of trips but lets a third of "on time" trips run late.
- **0.40** sits between them and moved very little from validation to test.

**For the product team:** no threshold makes "Usually on time" a strong promise. Even at
0.30, about one in five such trips runs late. We recommend copy that describes likelihood
("Usually on time: about 3 in 4 trips like this one leave within 3 minutes") rather than a
guarantee.

The spec's three levels need a second cut. We propose **0.60** for "Often late" (about 72%
of those trips ran late on validation's school week), leaving 0.40–0.60 as "Sometimes late".
The warning cap (spec §7) is still a product decision. If it is lower than about 60% of
trips, the threshold has to rise, and more late trips will be shown as on time.

## 6. Reproducibility

| Item | Value |
|---|---|
| Git commit | the commit that adds this section. The training run logged `32b0200-dirty` (trained before the Task 3 code was committed). Retraining with the committed code reproduces it exactly (below). |
| DVC splits (md5) | train `9c76528a5a338970f93ee5e6e3928ce2`, val `e381cfc557b368f1f9360888e12f3d02`, test `b58c1d86758c22e0863529294b79a3cc` |
| MLflow tracking | `sqlite:///mlflow.db`, artifacts in `mlruns/` |
| MLflow experiment | `pulse-late-3min` |
| Winning run | `logreg_history_rates`, run id `10194bd134054df4bef5ef658e3dfff3` |
| Registered model | `pulse-latenet` **version 1**, stage **Production**, alias `production` |
| Model version tags | threshold 0.4, recalibration rolling 7-day, test AUC 0.7375 |

Reproduce from the frozen splits (`dvc pull` or `python -m src.make_splits`):

```bash
python -m src.train configs/logreg_history_rates.yaml
python -m src.calibrate logreg_history_rates
python -m src.evaluate logreg_history_rates --threshold 0.40 --register
```

1. `src.train` builds the history features, trains on train, scores validation and logs the
   run and model.
2. `src.calibrate` applies the rolling recalibration to the validation predictions and logs
   the calibration metrics and threshold table.
3. `src.evaluate` is the only script that reads the test split. It refuses to run twice for
   the same run (pass `--force` only to deliberately re-evaluate). It then registers and
   promotes the model if it beats every baseline on test.

**Checked on 2026-11-18:** retraining `configs/logreg_history_rates.yaml` with the current code,
without logging, gave validation AUC 0.725536 and accuracy 0.662159, identical to the logged
run. lbfgs stops at its 300-iteration cap (a convergence warning, as in Task 2). The fit is
deterministic, so the result reproduces exactly; raising `max_iter` is a possible next tweak.

Load the production model with `mlflow.sklearn.load_model("models:/pulse-latenet@production")`.
It expects the feature columns built by `src.train.load_frames`. The rolling recalibration
and threshold are applied outside it.

## 7. Limitations & Next Steps

### What the model can't do

- **React quickly to change.** The recalibration lags a week, and the late rates lag up to
  eight weeks. The school-term morning peak was still under-predicted by 11–12 points
  in the second half of September.
- **Treat weekends separately from weekdays in recalibration.** The network-wide daily shift
  over-predicts weekends (49% predicted vs 31% actual on 26 September).
- **Predict day-specific disruption.** Weather, incidents, detours and events aren't in the
  features. That's most of the variation left above AUC 0.74.
- **Cover every stop.** Only timepoints have labels.
- **Serve as-is.** Production needs yesterday's actual departures to update the late rates
  and the recalibration. Today the pipeline gets actuals only from the monthly files; the
  daily API pull stores schedules only.
- **Count cancellations.** They are dropped with missing records, not counted as late.
- **Know the seasons.** There are only three months of history: one summer and two weeks of
  term, no winter.

### With more time

1. **Day-type and hour-band recalibration:** separate rolling shifts for weekdays and
   weekends and for the AM peak. This directly targets the two biggest calibration errors
   above.
2. **School calendar** once training spans a term, and planned service changes from
   `/alerts`.
3. **A daily actuals feed** (archived vehicle positions, used only for *earlier* trips) so the
   features and recalibration are truly "up to yesterday" in production.
4. **More history** (12+ months) and retraining on a rolling window, so the model has seen
   seasonal shifts instead of only reacting to them.
5. **A small neural network** with route and stop embeddings, the deferred architecture
   comparison. Given that trees didn't gain over logistic regression once history was in,
   expect a small gain.
6. **An unseen-routes check** (spec §5) to measure cold-start performance for new routes.

### If "late" meant five minutes instead of three

- **The classes would become imbalanced.** The median delay is 2.4 minutes and the 90th
  percentile 10.3, so a 5-minute cut would make late trips a clear minority instead of half.
  "Always on time" would then score well on accuracy, so accuracy would become misleading,
  and PR-AUC and recall would be the numbers to watch.
- **The signal would shift toward structure and rare events.** Delays over 5 minutes
  concentrate on chronically bad routes, peak hours and disruptions. Ranking could get
  easier for the structural part (the worst slots stand out more), but each slot would see
  fewer late events, so its recent late rate would be noisier. That calls for stronger
  shrinkage or longer windows, which in turn react more slowly to drift.
- **Warnings would become rarer and more meaningful.** The product could warn on a much
  smaller share of trips, and "Usually on time" would be a stronger promise. The threshold
  would sit much lower in probability, and calibration at low probabilities would matter
  most.
- **The rider promise would change.** A 4-minute-late bus would count as on time, which
  matches what riders tolerate better on frequent routes but worse for tight transfers.
  That's a product choice as much as a modelling one.