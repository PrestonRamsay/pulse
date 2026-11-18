# ML Spec: Pulse Reliability Signal
## 1. Problem statement
Riders using pulse rely on scheduled trips, but delays make it feel untrustworthy. Because complaints around departure times are the biggest category of complaint, we want to show how likely a trip is to be late for saved trips.
## 2. Prediction task
Unit: one departure from a timepoint stop: route, direction, timepoint stop, scheduled departure time and day type (see section 3 for why timepoints only).
Target: late = the departure is strictly more than 3 minutes after its scheduled time. The threshold is fixed at 3 minutes; the earlier 5-minute fallback is dropped.
Task Type: Binary classification. The output is a probability, shown as one of three levels: Usually on time, Sometimes late or Often late.
## 3. Features available at prediction time
List each feature, when it becomes known, and which source it comes from.
Revised 2026-10-30 against the real data (MBTA Bus Arrival/Departure files, Jul–Sep 2026, and the V3 API).

**The unit is a timepoint, not every stop.** The historical files only record timepoint stops (9 per direction on route 1; 2,485 route/direction/stop combinations across the network). Labels, and therefore predictions, exist only for those. Non-timepoint stops would need a separate method.

Schedule features (known when the schedule is published):
- Route, direction, stop, scheduled time. Source: historical files for training; MBTA V3 `/schedules` for serving, joined to the historical timepoints (the API's own `timepoint` flag only marks the first stop of each trip, so it can't be used to pick them).
- Hour of day and minute of day, from the scheduled time.
- Day type, from the **service date's schedule pattern**, not the weekday: Labor Day (Mon 7 Sep) ran the Sunday schedule (58k rows, against about 91k on a normal weekday). Trips after midnight belong to the previous service date.
- Timepoint position (`point_type`: start/mid/end, `time_point_order`). Source: historical files only; the API has `stop_sequence` over every stop but not timepoint order. These vary by trip pattern for 8% (point type) and 28% (order) of route/direction/stop combinations, so serving looks them up per trip pattern and time of day, not per stop. Late rate differs sharply by position: start 26%, midpoint 50%, end 36%.
- Schedule standard (`standard_type`: Headway or Schedule) and `scheduled_headway`. Source: historical files only, not in the API. `standard_type` changes with time of day on 35% of combinations; `scheduled_headway` is blank on 2.6% of rows, all of them Schedule-type.
- *Dropped:* distance from route origin. Neither source carries it; timepoint order and `stop_sequence` stand in for it.

History features (known the previous day; source: historical actuals in `stop_events`):
- This slot's late rate over the last 2, 4 and 8 weeks, and the same for route and hour where the slot has little history. These matter more than first thought: the late rate moved from 42% in July–August to 52% in the second week of September, and a model trained on summer under-predicted September. Route late rates range from 23% to 69%.
- 5% of historical rows have no actual time; they are excluded from these rates, not counted as on time.
- **As built (Task 3, `src/features.py`):** late rates at four levels (network; route; route/direction/stop; slot = route/direction/stop/hour/day type) over 7, 14, 28 and 56 service days. Each row uses only service days strictly before its own, in every split; its own label and its own day's labels never reach it (tested). Rates are shrunk toward the next-coarser level with a pseudo-count of 20, so thin slots fall back to their stop, route or network rate, and to 0.5 before any history exists. Train drops its first 14 days, whose windows are nearly empty.
- *Tried and dropped:* trip counts behind each rate. They grew as the windows filled from 1 July, so September looked unlike any training day and the model over-predicted (mean 0.57 against 0.49 actual).
- **What the model actually uses:** the slot's 8-week rate dominates (permutation importance 0.167 AUC; the next feature, route, 0.008). Once it is known, the schedule adds little.

Day type as built: weekday/Saturday/Sunday from the service date, with holidays mapped to the pattern they ran (3 July Saturday, 4 July and 7 September Sunday), found from daily trip counts. `day_of_week` is kept alongside it.

School calendar as built: **not a feature.** Every training day is summer, so a school-in-session flag would be constant in train and the model could learn nothing from it. The recent late rates carry the September change instead. It becomes usable once training spans a school term.

Recalibration (serving step, not a feature): each day's predictions are shifted on the log-odds scale so the previous 7 days' predictions would have matched those days' actual late rate. Same "earlier days only" rule as the history features. A one-off calibration fitted on 1–7 September did not carry into the school weeks; the rolling one cut calibration error on 8–15 September from 0.074 to 0.052.

Calendar and service changes (known ahead of time):
- Holidays, school calendar, schedule-change weeks. Source: calendar. The September jump lines up with schools returning, so the school calendar is a first-class feature, not optional.
- Planned detours and service changes. Source: MBTA V3 `/alerts`, using only alerts whose `created_at` is before the trip.
- Weather forecast (optional): known the day before. Source: weather provider.

Never features (recorded during or after the trip): `actual`, `delay_min`, observed `headway`, and anything from `/predictions` or `/vehicles`.
## 4. Label source
Where historical answers come from; how a row labels itself.
Historical actual departure times come from the MBTA Bus Arrival/Departure files (loaded into `stop_events` with `source = 'historical'`). The daily pull from the MBTA V3 API (`source = 'api'`) stores scheduled service only, with `actual` and `late` left empty, so it gives serving-side inputs but no labels; new labels arrive only when the next monthly historical file is loaded. There is no GTFS-Realtime archive. Each observed departure is matched to its scheduled trip, and the row labels itself: strictly more than 3 minutes after schedule is late (1), otherwise on time (0). Rows with delays outside −30 to +120 minutes are dropped as bad records.
Departures with no recorded time (about 5% of historical rows) are dropped rather than counted as on time. The files don't separate cancelled trips from missing records, so cancellations are currently dropped with them, not counted as late (see open questions).
History on hand: July–September 2026 (3 months, 7.53M rows). The original goal of at least 12 months, to capture a full seasonal cycle, still stands; until then the school calendar and recent late-rate features carry the seasonal signal.
## 5. Metrics, baseline, and target
Baselines, from simplest to the bar the model must clear:
- Majority class (always "on time"). Built: 51.3% accuracy, AUC 0.500, PR-AUC 0.487 on validation.
- Logistic regression on schedule features only. Built: 62.6% accuracy, AUC 0.681, PR-AUC 0.652 on validation.
- One network-wide late rate for every trip, last 8 weeks. Built (`history_network_8wk`): AUC 0.518.
- Each slot's late rate over the last 8 weeks. Built (`history_slot_8wk`): 65.9% accuracy, AUC 0.723, PR-AUC 0.703. The model ships only if it beats this one.

Model chosen on validation (Task 3): `logreg_history_rates`, logistic regression on one-hot schedule fields plus the history rates, with rolling 7-day recalibration. Full validation: 66.2% accuracy, AUC 0.726, PR-AUC 0.706. On the 8–15 September school-week check, rolling recalibration reduced calibration error from 0.074 to 0.052 for logistic regression (Brier 0.218, AUC 0.718); for gradient-boosted trees it fell from 0.083 to 0.061 (Brier 0.220, AUC 0.712). The logistic regression is the selected model because it calibrates better and is slightly stronger on that school-week check. The margin over the 8-week slot baseline is small (0.003 AUC); the margin over the Task 2 logistic regression is 0.045. Registered as `pulse-latenet`.
Threshold: **0.40** on the recalibrated probability, selected and fixed on validation before the single test evaluation. Below it a trip shows "Usually on time". On validation it warned on 60% of trips, caught 76% of late ones, and 29% of "Usually on time" trips ran late. It leans pessimistic, as this section asks; the warning cap is still an open question.
Test result (single evaluation, 16–30 September): AUC 0.738, accuracy 66.2%, recall 78.2%, precision 61.8%, 27.0% of "Usually on time" trips late. It beats the 8-week slot baseline (AUC 0.732) by +0.006 (day-bootstrap 95% interval 0.0003 to 0.010), so it meets this section's ship rule, narrowly. Details in REPORT.md §4.
Target: set from the data. The starting goal is 70–80% accuracy, but accuracy can be misleading on its own. About 42–49% of trips run late, so predicting "on time" for everything already scores about 51%. Accuracy, AUC-ROC and PR-AUC are reported alongside the two numbers below.
Evaluation data, split by service date: train Jul 1 – Aug 31, validation Sep 1–15, test Sep 16–30 (held out until the final score). Splits are versioned in DVC. As more months load, the held-out window grows toward the most recent 4–8 weeks. A split of whole routes the model never saw is still planned, not built.
Expensive mistake: telling a rider a trip is reliable and then having it run late. We'll track:
- How often trips shown as "Usually on time" actually run late. This is the main number to minimize.
- What share of saved trips show a warning. This is capped so the warning still means something.
The cut points between levels are set to lean slightly pessimistic, within that cap. Neither number is computed yet; they need calibrated probabilities and chosen cut points first.
## 6. Non-goals
- Predicting arrival or departure times to the minute
- Trip routing
- Real-time disruption alerts (v1 reflects typical reliability, not today's incidents)
- Explaining why a trip runs late
## 7. Open questions
Should early departures count as unreliable?
Should cancelled trips count as late, and is there a source that identifies them (e.g. V3 `/schedules` trips with no departure record)?
What warning cap, and how much pessimism, is product comfortable with?
How should frequent routes be handled, where riders don't plan around the schedule?
Do support tickets include route and time, so we can measure impact?
## 8. Change log
Original Task 1 wording of the parts that changed, and why each changed:
- §2 unit, original: "one trip slot: route, direction, stop, scheduled departure time and day type." Changed because the files only record timepoints, so labels exist only there.
- §2 target, original: "more than 3 minutes late and if the system can't handle that then we will try 5 minutes." Changed because the graded spec fixes 3 minutes; at 3 minutes the classes are nearly balanced (42–49% late), so there was no need to fall back to 5.
- §4, original: "Pulse's own GTFS-Realtime archive fills the gaps." No such archive exists; the daily API pull stores schedules only. Original: "Cancelled trips count as late." The files don't separate cancellations from missing records, so they are dropped with them for now (open question). Original: "We need at least 12 months of history." Still the goal; only 3 months are loaded.
- §5 baseline, original: "each slot's late rate over the last 8 weeks … one network-wide late rate." Kept, and built in Task 3; the majority-class and schedule-only logistic regression baselines were added in Task 2 as simpler floors.
- §5 target example, original: "If about a quarter of trips run late, predicting 'on time' for everything already scores around 75%." The real late rate is 42–49%, so that model scores about 51%.
- §5 evaluation, original: "the most recent 4–8 weeks, held out from training, plus whole routes the model never saw." Built as two date-based weeks each for validation and test; the unseen-routes check is not built.
- §3 school calendar, original: "a first-class feature, not optional." Not used: constant in a summer-only training set (see §3).
- §7 removed as answered: "Is the service area MBTA only?" (yes) and "Do the historical actuals cover every stop, or only timepoints?" (timepoints only).

Dated entries:
- 2026-10-30: Section 3 revised against Jul–Sep 2026 data: timepoint-only unit, day type from service pattern, schedule fields the API lacks, distance from origin dropped, observed late rates and drift, explicit never-features list.
- 2026-11-16: Sections 2, 4, 5 and 7 brought in line with what's built: timepoint unit and fixed 3-minute threshold; label source is the historical files only (the daily V3 pull stores schedules, not actuals), with cancellations currently dropped; 3 months of history on hand; built baselines and their scores; real late rate (42–49%); date-based train/val/test splits; answered questions removed (MBTA only, timepoints only); cancellation question added.
- 2026-11-16 (Task 3): history features and day type as built, count features dropped, school calendar not used, rolling recalibration added, 8-week baselines built, model chosen on validation; original Task 1 wording kept above.
- 2026-11-16 (Task 3 handoff): recorded the 8–15 September school-week comparison after rolling recalibration, confirmed `logreg_history_rates` as the validation choice, and fixed the rider warning threshold at 0.40 before the still-pending single test evaluation.
