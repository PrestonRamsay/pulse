# Pulse Model Delivery Report

## 1. Data

### Months used

**July, August and September 2026**, the three most recent consecutive months published in the
2026 MBTA Bus Arrival/Departure zip (October was not yet available on 2026-10-29). They are loaded
into the Postgres `stop_events` table by `python -m src.load_historical`, and they are the
only months in the frozen train/validation/test splits.

What else is in the table, and why "three months" and "four months" were both said:
- **October 2026:** two days of `api` rows (29–30 October). These are schedules from the daily
  API pull, with no actual times. So four months had rows, but only three had labels.
- **April–June 2026:** loaded on 2026-11-24 from the same zip (7.43M rows), for the follow-up
  experiments in §3. They are kept out of the splits and used through `data/splits/apr_jun.parquet`.
- **18–23 November 2026:** `lamp` rows, actual times from MBTA LAMP (third source, below).

January–March 2026 are also in the zip but not loaded.

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

The API only gives schedules. `/schedules` accepts past dates but returns planned times;
`/predictions` rejects a date filter ("Unsupported filter(s): date"); and `/vehicles` is live
positions only. So the API can't say how late yesterday's buses were.

### Third source: LAMP daily actual times

MBTA's LAMP project publishes bus events with **TransitMaster actual times**, the same system
behind the monthly files, at
`https://performancedata.mbta.com/lamp/tableau/bus/`. There are two files:
- `LAMP_RECENT_Bus_Events.parquet`: about 175 MB, the last 7 service days including yesterday,
  refreshed every morning.
- `LAMP_ALL_Bus_Events.parquet`: about 8.8 GB, from 31 December 2025, read remotely by row
  group, about 50 MB per month.

Actual times exist only at timepoints, the same stops this project models.

**Agreement with the monthly files** (`python -m src.lamp_check`, 1–15 September, the
validation weeks; no model involved):

| Check | Result |
|---|---:|
| Monthly-file departures matched in LAMP | 1,125,842 of 1,155,247 (97.5%) |
| Actual time identical to the second | 99.80% (99.83% within 60 s) |
| Late label agrees | **99.92%** (start 99.89%, mid 99.95%, end 99.76%) |
| Late rate on matched rows | monthly 49.12%, LAMP 49.16% |

The unmatched 2.5% are mostly Silver Line trips and routes 712/713. Getting there needed four
conventions, found by matching the two sources (`src/lamp.py`):

- **Times:** use LAMP's seconds-after-midnight columns on the service date. Its planned-time
  timestamp column is shifted early by the UTC offset (4 hours in September, 5 in November),
  so it can't be used. A first attempt that ignored this matched only 6.5% of rows, and those
  agreed on the label only 58% of the time.
- **Route ids:** single digits are zero-padded in the files (`1` becomes `01`). The Silver
  Line, CT3 and 194 use GTFS ids in LAMP: SL1 = 741, SL2 = 742, SL3 = 743, SL4 = 751, SL5 = 749,
  SLW = 746, CT3 = 708, 194 = 8993. Each pair matched 99.9–100% of that route's departures.
  School variants 191–193 and `746_` appear under their main route.
- **Direction:** 0 = Outbound, 1 = Inbound, as in the API.
- **Endpoints:** the monthly files record the *arrival* at a route's last stop. LAMP's arrival
  matches it to the second 99.3% of the time there; its departure matches only 68.9%.

**The daily pull.** `python -m src.pull_lamp` downloads the recent file each morning and loads
every finished service day into `stop_events` with `source = 'lamp'`. Today is skipped while
it's still running. The first run (24 November) loaded 466,054 rows for 18–23 November, about
86,000 per weekday, with 41–53% late. Reruns update the same rows and insert nothing.
`--start/--end` backfills from the big file.

Sources don't overwrite each other blindly. For the same key, **monthly file > LAMP > API**:
- LAMP never touches a monthly-file row. A trial upsert of 1 September, rolled back, left all
  90,667 monthly rows and their actual times unchanged.
- A monthly file replaces LAMP rows for its days when it arrives.
- LAMP fills in actual times on schedule-only API rows.

Training splits read `source = 'historical'` only, so the frozen splits never change.

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
`configs/<run name>.yaml`; the current leader's is also saved as `configs/final.yaml`
(identical settings to `configs/logreg_history_rates.yaml`). The run table, best validation
AUC first, is exported to `mlflow_runs.csv`. There are 20 runs in all: the 10 below and the
10 neural-network runs further down.

> **Status (2026-11-24): the final model choice is open.** The logistic regression
> `logreg_history_rates` leads on validation and is the version currently in Production,
> but the choice is made last, on validation, after four checks:
> 1. **Serving:** the daily API pull has no actual times, so yesterday's outcomes aren't known
>    from it. Are the history features and recalibration available at serving time, and at
>    what lag? *Answered (§1, third source):* MBTA's LAMP publishes yesterday's actual times
>    every morning. It agrees with the monthly files on 99.92% of late labels, and
>    `src/pull_lamp.py` loads it daily, so "up to yesterday" is available as designed.
> 2. **Window meaning:** labelled data starts on 1 July, so a "56-day" rate in July covers
>    only a few weeks of summer. In September it covers 56 days that include school weeks.
>    *Answered (experiment A, below):* filling the windows with April–June changed validation
>    AUC by +0.0002 (logistic regression) and −0.006 (net). It was a real flaw, but not the cause.
> 3. **Seed noise:** each net config ran with one seed. Is the 0.010 gap with history, or the
>    0.025 gap without it, bigger than seed-to-seed variation? *Answered (2026-11-24, clean
>    commit `0f4eec4`):* the April–August net with seeds 0–3 reached best validation AUC
>    0.7347, 0.7363, 0.7356 and 0.7353 (mean 0.7355, standard deviation 0.0007). Its lead
>    over the logistic regression (0.0031) is more than twice the seed spread (0.0013), so the
>    ranking advantage is real. But seeds 1–3 peaked at **epoch 1**, with AUC falling 0.0004–0.0022
>    by the last epoch. Under the rule below, decided before these results, the net fails
>    the stability condition. **The logistic regression (April–August) is the final model.**
> 4. **More months:** which months help September? *Answered (experiment B, below):*
>    school-term months. Training on April–August lifts the logistic regression by 0.007 and
>    the net by 0.018, and fixes the net's epoch-1 peak. January–March, more term weeks plus
>    winter, is the next candidate.
>
> **Decisions fixed on 2026-11-24, before the seed results:**
> - **Training window:** 15 April – 31 August (experiment B). It beat July–August on validation
>   for both model families, so this doesn't depend on which model wins.
> - **Selection rule:** the net (April–August, seeds 0–3) is the final model only if **both**
>   hold:
>   1. its mean validation AUC beats the logistic regression's 0.7324 by more than **twice
>      the standard deviation** across the four seeds;
>   2. it trains stably in **every** seed: best epoch later than epoch 1, and last-epoch
>      validation AUC within 0.005 of its best (no repeat of the old net's decline after
>      epoch 1).
>
>   Otherwise the logistic regression (April–August) is the final model, on simplicity and
>   calibration. The logistic regression is deterministic, so it needs no seeds.
> - **Then, on validation and before any test scoring:** re-pick the threshold for the final
>   model, and apply the same rolling 7-day recalibration whichever model wins.
>
> These use train and validation only. Test is not touched again without the reviewer's OK (§4).

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

### Neural network (latenet)

A small feed-forward network (`src/latenet_model.py`, trained by `src/train_latenet.py`):

- **Inputs:** learned embeddings for route (16 dimensions), stop (32), hour (8), day of week
  (4) and point type (2). Vocabularies come from train only, with index 0 for unseen values.
  Numeric inputs are direction, weekend, standard type and scaled timepoint order.
- **Model:** ReLU hidden layers with dropout and one logit output, trained with Adam and
  binary cross-entropy.
- **Training:** each epoch is logged to MLflow (`step=epoch`). Early stopping on validation
  AUC (patience 3) keeps the best epoch's weights.

The sweep changed one setting at a time from `latenet_base`:

| Run | Change | Best val AUC | Best epoch |
|---|---|---:|---:|
| `latenet_base` | hidden [128, 64], dropout 0.1, lr 0.001, batch 8,192 | 0.7055 | 11 |
| `latenet_wide` | hidden [256, 128] | 0.7080 | 11 |
| `latenet_lr003` | learning rate 0.003 | 0.7078 | 7 |
| `latenet_deep` | hidden [128, 64, 32] | 0.7069 | 13 |
| `latenet_batch32k` | batch 32,768 | 0.7040 | 12 |
| `latenet_dropout03` | dropout 0.3 | 0.7035 | 9 |
| `latenet_emb_small` | embeddings halved | 0.7009 | 7 |
| `latenet_wide_lr003` | wide + lr 0.003 | 0.7081 | 7 |
| `latenet_history` | wide + lr 0.003 + the 16 late rates (log-odds, standardized on train) | 0.7163 | 1 |
| **`latenet_final`** | `latenet_history`, rerun from a clean commit | **0.7163** | **1** |

- **The sweep was flat.** Every change moved AUC by less than 0.005. Width and a higher
  learning rate helped a little, and smaller embeddings, heavier dropout and bigger batches
  hurt a little. No run hit the 20-epoch cap.
- **On schedule fields alone, the net is the best model** (0.708 against 0.706 for trees and
  0.683 for logistic regression). Embeddings learn route, stop and hour interactions without
  hand-built features.
- **History added 0.008 AUC, but the net peaked in epoch 1.** After that, validation AUC fell
  every epoch while train loss kept falling. In September the history inputs sit above
  anything seen in summer (the network rates are 2.5–3 train standard deviations high). The
  more closely the net fits summer, the worse it extrapolates. It also under-predicts more
  than the logistic regression: its mean prediction is 0.395 against 0.487 actual. The logistic
  regression is linear in the same inputs, so it extrapolates smoothly.
- **Why it trails for now:** it scored 0.010 AUC below `logreg_history_rates` on validation.
  It is less stable (best at epoch 1), 70 times larger (93,649 parameters against 1,357) and
  less well calibrated. Without history inputs the order flips: the net leads by 0.025.
  So if check 1 above removes the history features, the comparison changes. `latenet_final`
  was scored on test once, after the logistic regression's test result was known (§4).

### Follow-up experiments after review (validation only)

Two experiments test why both model families learned summer and struggled with September.
Both use April–June 2026, loaded on 2026-11-24 and frozen as `data/splits/apr_jun.parquet`
(7,054,235 rows, 45.8% late, DVC md5 `05c9b891…`, built by `src/make_extra.py`). Validation
(1–15 September) and test are unchanged, and test was not read.

- **A, full windows (`extra_data.use: history`):** April–June only feeds the late-rate
  windows. The training rows are the same as before (15 July – 31 August), but every July
  row now has a genuine 56 days of history instead of a few weeks (the window-meaning
  question, status point 2).
- **B, longer training (`extra_data.use: train`):** train on 15 April – 31 August (14-day
  burn-in), so the models see school-term weeks. May ran 48.0% late, almost the same as
  September's 48.5%. July and August ran about 42.5%.

| Run | Train days | Train rows | Val AUC | Change | Val accuracy (0.5) | Notes |
|---|---|---:|---:|---:|---:|---|
| `logreg_history_rates` (current leader) | 15 Jul – 31 Aug | 3.72M | 0.7255 | n/a | 66.2% | mean prediction 0.449, Brier 0.214 |
| A: `logreg_history_rates_fullwin` | 15 Jul – 31 Aug | 3.72M | 0.7257 | +0.000 | 66.2% | mean prediction 0.444 |
| **B: `logreg_history_rates_apr_aug`** | 15 Apr – 31 Aug | 10.74M | **0.7324** | **+0.007** | 66.9% | mean prediction 0.468, Brier 0.210, PR-AUC 0.713 |
| `latenet_final` | 15 Jul – 31 Aug | 3.72M | 0.7163 | n/a | 64.4% | best epoch 1, then worse every epoch |
| A: `latenet_history_fullwin` | 15 Jul – 31 Aug | 3.72M | 0.7105 | −0.006 | 62.6% | best epoch 1 |
| **B: `latenet_history_apr_aug`** | 15 Apr – 31 Aug | 10.74M | **0.7347** | **+0.018** | 67.1% | best epoch 5; val loss 0.648 → 0.607 |

The validation late rate is 48.7%. Net accuracies are at the best epoch.

- **A changed nothing that matters.** Truncated windows were a real flaw in what "56-day rate"
  meant during training. But filling them moved the logistic regression by 0.0002 and the net
  by −0.006, which is within the sweep's seed-level spread. The rates are shrunk toward their
  parent level, so short windows already behaved sensibly.
- **B is the fix.** Seeing school-term weeks lifts both models, and the net gains 2.5 times
  as much. The net's training curve is also healthy now: validation AUC stays at 0.733–0.735
  across all 8 epochs instead of peaking in epoch 1. Its probabilities are much closer to
  reality (validation loss 0.648 to 0.607), and its accuracy at 0.5 rises from 64.4% to 67.1%.
  This matches the diagnosis above: the net's problem was extrapolating from a summer-only
  world, and the logistic regression's straight lines were less exposed to that.
- **The ranking flips, narrowly:** net 0.7347, logistic regression 0.7324. The gap (0.002) is
  smaller than the 0.005 spread of the original sweep, so it is being checked with seeds 1–3
  of the B net (status point 3) before the final choice.
- **Provenance:** these four runs logged `a43d170-dirty`, because the code was committed
  afterwards in `5f57c95`. Run ids:
  - `logreg_history_rates_fullwin` `2e992a27…`
  - `logreg_history_rates_apr_aug` `145444ad…`
  - `latenet_history_fullwin` `4ff34bc9…`
  - `latenet_history_apr_aug` `3ddab843…`

  The seed runs and the final winner are trained from a clean commit.

### Current leading architecture

`logreg_history_rates`, registered as `pulse-latenet` v1 (final choice pending, see the
status note above):

- **Inputs:** one-hot route, direction, stop, point type, standard type, hour, day of week
  and day type (1,338 columns, `handle_unknown="ignore"`), plus 18 standardized numeric
  features: scheduled headway, timepoint order, and the 16 late rates on the log-odds scale.
- **Model:** L2-regularized logistic regression (lbfgs, C = 1.0, max_iter 300).
  **1,357 parameters** (1,356 weights + 1 intercept).
- **Serving step:** rolling 7-day recalibration (one log-odds shift per day, computed from
  the previous week's predictions and outcomes), then a 0.40 warning threshold (§5).

It ties the trees on AUC, is better calibrated before and after recalibration, and each
weight can be read directly. That's why it leads `gbt_history`. It beats the neural
network by 0.010 AUC on validation at a seventieth of the size.

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

**When test was used, and what that means.** Test has been scored twice, once per model:

| Date | Model | Why |
|---|---|---|
| 2026-11-18 | `logreg_history_rates` (logistic regression) | the single final evaluation of the Task 3 choice |
| 2026-11-23 | `latenet_final` (neural network) | the latenet brief's final evaluation, run **after** the first test result was known |

Neither model was trained on test, and neither threshold was moved after seeing it. But
once a test result is known, it can influence later decisions, even unintentionally. That
applies especially to the second evaluation, and to any choice between the two models. So
the test numbers below are **no longer a fully clean estimate** of performance on unseen
weeks. They stay as recorded. Further work uses train and validation only, the final model
will be chosen on validation, and test won't be scored again without checking with the
reviewer first. A genuinely fresh check would use newer data, such as October 2026, once its
monthly file is loaded.

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
Across all 122 routes with at least 2,000 test rows, the Pearson correlation between
predicted and actual route late rates is **0.916** (0.903 over the 78 routes with at least
5,000 rows). So the ranking holds at route level as well as trip level. This was computed
from the saved test predictions (`data/preds/logreg_history_rates_test.parquet`); the model was
not re-scored.

### The neural network on test (scored once, after the logistic regression's test run)

`latenet_final` (MLflow run `1ac5d213…`) was scored once on test with
`python -m src.evaluate_latenet test --threshold 0.35 --run-id <run id>`, after the logistic
regression's test result was known. The 0.35 threshold was fixed on validation beforehand. The net has
no recalibration step. Its metrics are logged on its run.

| | Validation | Test |
|---|---:|---:|
| AUC-ROC | 0.716 | 0.742 |
| PR-AUC | 0.696 | 0.715 |
| Accuracy at 0.5 | 64.4% | **62.8%** |
| Accuracy at 0.35 | 65.6% | 67.1% |
| Recall at 0.35 | 68.4% | 54.9% |
| Share flagged at 0.35 | 52.3% | 37.6% |
| "On time" but late at 0.35 | 32.3% | 35.0% |
| Mean predicted vs. actual late rate | 0.395 vs. 0.487 | **0.311 vs. 0.483** |
| Route Pearson r (≥ 5,000 rows) | 0.928 | 0.923 |

Its **ranking held**: test AUC 0.742 is slightly above the logistic regression's 0.738, and
route correlation stayed at 0.92. But its **probability level collapsed**. By late September
the history inputs are even further outside the summer range, and instead of following them
up, the net's output fell. It under-predicts in every hour of the day by 10–27 points, worst
at 8am (28% predicted against 55% actual). So a fixed threshold flags fewer buses and catches
fewer late ones. It would miss the 65% accuracy bar at 0.5, while the logistic regression
clears it (67.3%). Rolling recalibration would most likely fix the level, since the ranking is intact,
but it was not tried on test, so as not to tune on test.

### Serving latency

`python -m src.latency_check` loads the DVC-tracked model (`models/pulse-latenet`), builds
one real validation departure's inputs, and times 200 single-row predictions on CPU after 10
warm-up calls. Each call is `predict_proba`, then the recalibration shift, then the threshold.

| Measured 2026-11-23 | |
|---|---|
| CPU | Intel64 Family 6 Model 191 (laptop), Windows 11, CPU only |
| Single prediction | **median 12.8 ms**, p95 14.2 ms, max 17.0 ms |
| Model load | 2.9 s, once at startup |

**It passes the 200 ms bar** with a wide margin. The late rates and the day's
recalibration shift are computed in a daily batch and looked up at serving time, so they
aren't in the per-request time (building them for one row from the raw splits takes about
5 s).

### Against the acceptance bars

- **Model quality:** test accuracy at 0.5 is **67.3%** (bar 65%), and AUC is **0.738** (bar
  0.70). That's 15.6 points above the test weeks' majority baseline (51.7%). At the 0.40
  operating threshold, accuracy is 66.2%, traded for 78% recall.
- **Serving-safe features and time-based splits:** every feature is known the day before (the
  late rates use only earlier service days, and this is tested). Splits are by service date.
  Each candidate model was scored on test exactly once: the logistic regression on
  2026-11-18, and the net on 2026-11-23 for comparison.
- **Latency:** 12.8 ms median (bar 200 ms).

## 5. Threshold Recommendation

### Final model (April–August logistic regression): 0.40, fixed on validation on 2026-11-24

The final model (`configs/final.yaml`, MLflow run `4235459a…`, clean commit `936714b`) was
retrained on 15 April – 31 August. That reproduced experiment B exactly: validation AUC
0.7324 and accuracy 66.9% at 0.5. Its threshold was re-picked from its own recalibrated
validation probabilities with `python -m src.calibrate final`, **before any test scoring**,
and the choice is tagged on the run (`chosen_threshold = 0.40`).

| Threshold | Late buses caught | On-time buses flagged | All buses flagged | "Usually on time" but late | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 83.6% | 54.4% | 68.6% | 25.4% | 64.1% |
| **0.40** | **78.0%** | **46.2%** | **61.7%** | **28.0%** | **65.6%** |
| 0.45 | 71.8% | 38.4% | 54.7% | 30.3% | 66.6% |
| 0.50 | 65.0% | 31.3% | 47.7% | 32.6% | 66.9% |

All of validation (1–15 September, 48.7% late), after rolling 7-day recalibration. In the
school weeks (8–15 September), 0.40 catches 79.1% and flags 64.5% of buses.

- **0.40 (recommended):** we catch 78% of late buses at the cost of flagging 46% of on-time
  ones. It leans pessimistic, as the spec's expensive mistake asks, and warns on about 62% of
  trips, in line with v1's operating point (59.5%).
- **0.35:** catches 84%, but flags more than half of on-time buses and warns on 69% of trips,
  while only 49% run late, so the warning starts to lose meaning.
- **0.45 and 0.50:** fewer, sharper warnings (55% and 48% of trips), but 3 in 10 trips shown as
  "Usually on time" run late, and a quarter to a third of late buses get no warning.

Recalibration nudges validation AUC from 0.7324 to 0.7309, because each day gets its own
shift. It brings the mean prediction from 0.468 to 0.484, against 0.487 actual.

### Task 3 model (v1, July–August), for the record

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

In product terms, on validation:

- **At 0.40 (recommended), we catch 76% of late buses at the cost of flagging 44% of on-time
  ones.** 29% of trips shown as "Usually on time" still run late.
- **At 0.30, we catch 87% of late buses but flag 60% of on-time ones.** This protects riders
  most (24% of "on time" trips late), but it warns on three trips in four, which risks
  warnings being ignored.
- **At 0.50, we catch 63% of late buses and flag 30% of on-time ones.** It warns on fewer than
  half of trips but lets a third of "on time" trips run late.

On test, the same thresholds gave 78% / 45% at 0.40, 89% / 61% at 0.30 and 65% / 31% at 0.50
(late buses caught / on-time buses flagged). Every row moved by 2 points or less. The
intermediate rows: 0.35 caught 81% and flagged 52% on validation; 0.45 caught 70% and flagged
37%.

**What moved from validation to test:** very little at 0.40. The warned share went from
59.5% to 61.1%, recall from 75.7% to 78.2%, and "on time" but late fell from 29.3% to 27.0%.
That stability comes from the rolling recalibration. The un-recalibrated neural network shows
what happens without it: its share flagged at a fixed 0.35 fell from 52% to 38% (§4).

**How often to re-pick:** the threshold stays at 0.40 for this release; it was not moved
after the test results. We propose re-picking it **monthly**, when each new month of labelled
files arrives, on the most recent two labelled weeks. Recalibration stays daily. The
threshold is re-picked in the same way it was chosen here (pessimistic lean, within the warning
cap), and the new value is logged on the registered model version as a tag.

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
| Git commit | Code, configs and this report: the commit that adds `configs/final.yaml` with the logistic regression (`git log -1 -- configs/final.yaml`). The leading run logged `32b0200-dirty` (it was trained before the Task 3 code was committed). Retraining with the committed code reproduces it exactly (below). |
| DVC splits (md5) | train `9c76528a5a338970f93ee5e6e3928ce2`, val `e381cfc557b368f1f9360888e12f3d02`, test `b58c1d86758c22e0863529294b79a3cc` |
| DVC model | `models/pulse-latenet` (the MLflow model of version 1), md5 `22190144455cc3bf4750c8794b8687b8.dir`. The neural network's weights are also tracked: `models/latenet.pt`, md5 `fcde379666e7ae31a779328122b4c4ea`. |
| DVC remotes | default `github`: the public store `PrestonRamsay/pulse-dvc-store`, read over HTTPS with no account. A plain `dvc pull` from an empty folder fetched all 12 files with matching md5s (2026-11-24). Backup `origin`: DagsHub, login needed. |
| MLflow tracking | `sqlite:///mlflow.db`, artifacts in `mlruns/` |
| MLflow experiment | `pulse-late-3min` (20 runs) |
| Current leading run (final choice pending) | `logreg_history_rates`, run id `10194bd134054df4bef5ef658e3dfff3`, config `configs/final.yaml` |
| Registered model | `pulse-latenet` **version 1**, moved through **Staging** to **Production**, alias `production`. Its description names the source run. Screenshot of the registry page: [`assets/registry_pulse-latenet_production.jpg`](assets/registry_pulse-latenet_production.jpg) (2026-11-24). The final April–August model becomes version 2 after its test run, and the screenshot will be retaken then. |
| Model version tags | threshold 0.4, recalibration rolling 7-day, test AUC 0.7375 |

Reproduce from a fresh clone (README.md has the setup):

```bash
dvc pull                                   # frozen splits and the model
python -m src.train configs/final.yaml     # the leading run: validation AUC 0.7255
python -m src.calibrate logreg_history_rates
python -m src.evaluate logreg_history_rates --threshold 0.40   # test, once per run
python -m src.latency_check                # single-prediction latency
```

1. `src.train` builds the history features, trains on train, scores validation, and logs the
   run and model. `configs/final.yaml` keeps the run name `logreg_history_rates`, so the
   later steps find the fresh run.
2. `src.calibrate` applies the rolling recalibration to the validation predictions and logs
   the calibration metrics and threshold table.
3. `src.evaluate` is the only script that reads the test split. It refuses to run twice for
   the same run (pass `--force` only to deliberately re-evaluate). With `--register`, it also
   registers and promotes the model if it beats every baseline on test. A reproduction doesn't
   need that flag.

The neural network reproduces with
`python -m src.train_latenet configs/latenet_final.yaml`, which gives validation AUC 0.7163
at epoch 1. That run was retrained once to check this, and matched to four decimals.

**Checked on 2026-11-18:** retraining `configs/logreg_history_rates.yaml` with the current code,
without logging, gave validation AUC 0.725536 and accuracy 0.662159, identical to the logged
run. lbfgs stops at its 300-iteration cap (a convergence warning, as in Task 2). The fit is
deterministic, so the result reproduces exactly; raising `max_iter` is a possible next tweak.

Load the production model with `mlflow.sklearn.load_model("models:/pulse-latenet@production")`,
or from the DVC copy with `mlflow.sklearn.load_model("models/pulse-latenet")`.
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
- **Serve end to end yet.** Yesterday's actual departures now arrive daily from LAMP
  (`src/pull_lamp.py`, §1), but the morning job that turns them into late rates and the
  recalibration shift, then scores the day's schedule, isn't built. `pull_lamp` also has to be
  scheduled; so far it has been run by hand. It also depends on LAMP's undocumented
  conventions staying stable (§1), so `src/lamp_check.py` should be rerun whenever a monthly
  file arrives.
- **Count cancellations.** They are dropped with missing records, not counted as late.
- **Know the seasons.** There are only three months of history: one summer and two weeks of
  term, no winter.

### With more time

1. **Day-type and hour-band recalibration:** separate rolling shifts for weekdays and
   weekends and for the AM peak. This directly targets the two biggest calibration errors
   above.
2. **School calendar** once training spans a term, and planned service changes from
   `/alerts`.
3. **The daily scoring job** on top of the LAMP feed (now built and checked): each morning, load
   yesterday's LAMP actuals, update the late rates and the recalibration shift, and score
   today's schedule from the API pull.
4. **More history** (12+ months) and retraining on a rolling window, so the model has seen
   seasonal shifts instead of only reacting to them.
5. **Make the neural network serveable.** It ranks as well as the logistic regression (test
   AUC 0.742), but its probabilities sit far too low (§4). Next steps: give it the same
   rolling recalibration, train it on more than one season so the history inputs aren't
   out of range, and stop it training before it overfits summer (it peaked in epoch 1). Its
   real advantage, learning interactions, should pay off once there are new kinds of input
   to combine: weather, planned detours, school days.
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