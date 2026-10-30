# ML Spec: Pulse Reliability Signal
## 1. Problem statement
Riders using pulse rely on scheduled trips, but delays make it feel untrustworthy. Because complaints around departure times are the biggest category of complaint, we want to show how likely a trip is to be late for saved trips.
## 2. Prediction task
Unit: one trip slot: route, direction, stop, scheduled departure time and day type 
Target: we want to track departures that are more than 3 minutes late and if the system can't handle that then we will try 5 minutes.
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

Calendar and service changes (known ahead of time):
- Holidays, school calendar, schedule-change weeks. Source: calendar. The September jump lines up with schools returning, so the school calendar is a first-class feature, not optional.
- Planned detours and service changes. Source: MBTA V3 `/alerts`, using only alerts whose `created_at` is before the trip.
- Weather forecast (optional): known the day before. Source: weather provider.

Never features (recorded during or after the trip): `actual`, `delay_min`, observed `headway`, and anything from `/predictions` or `/vehicles`.
## 4. Label source
Where historical answers come from; how a row labels itself.
Historical actual departure times come from the agency's arrival and departure records. Pulse's own GTFS-Realtime archive fills the gaps. Each observed departure is matched to its scheduled trip, and the row labels itself: more than 3 minutes after schedule is late (1), otherwise on time (0). Cancelled trips count as late. Departures with no recorded time are dropped rather than counted as on time. We need at least 12 months of history to capture seasonal patterns.
## 5. Metrics, baseline, and target
Baseline: each slot's late rate over the last 8 weeks. We'll also score an even simpler version, one network-wide late rate for every trip, so we can see how much each step adds. The model ships only if it clearly beats the 8-week baseline.
Target: set from the data. The starting goal is 70–80% accuracy, but accuracy can be misleading on its own. If about a quarter of trips run late, predicting "on time" for everything already scores around 75%. Accuracy will be reported alongside the two numbers below.
Evaluation data: the most recent 4–8 weeks, held out from training, plus whole routes the model never saw.
Expensive mistake: telling a rider a trip is reliable and then having it run late. We'll track:
- How often trips shown as "Usually on time" actually run late. This is the main number to minimize.
- What share of saved trips show a warning. This is capped so the warning still means something.
The cut points between levels are set to lean slightly pessimistic, within that cap.
## 6. Non-goals
- Predicting arrival or departure times to the minute
- Trip routing
- Real-time disruption alerts (v1 reflects typical reliability, not today's incidents)
- Explaining why a trip runs late
## 7. Open questions
Is the service area MBTA only, or are other agencies included?
Do the historical actuals cover every stop, or only timepoints?
Should early departures count as unreliable?
What warning cap, and how much pessimism, is product comfortable with?
How should frequent routes be handled, where riders don't plan around the schedule?
Do support tickets include route and time, so we can measure impact?
## 8. Change log
- 2026-10-30: Section 3 revised against Jul–Sep 2026 data: timepoint-only unit, day type from service pattern, schedule fields the API lacks, distance from origin dropped, observed late rates and drift, explicit never-features list.
