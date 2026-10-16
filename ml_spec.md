# ML Spec: Pulse Reliability Signal
## 1. Problem statement
Riders using pulse rely on scheduled trips, but delays make it feel untrustworthy. Because complaints around departure times are the biggest category of complaint, we want to show how likely a trip is to be late for saved trips.
## 2. Prediction task
Unit: one trip slot: route, direction, stop, scheduled departure time and day type 
Target: we want to track departures that are more than 3 minutes late and if the system can't handle that then we will try 5 minutes.
Task Type: Binary classification. The output is a probability, shown as one of three levels: Usually on time, Sometimes late or Often late.
## 3. Features available at prediction time
List each feature, when it becomes known, and which source it comes from.
- Route, direction, stop, scheduled time, day type: known when the schedule is published. Source: GTFS schedule.
- Distance and stops from route origin, headway: known when the schedule is published. Source: GTFS schedule.
- This slot's late rate over the last 2, 4 and 8 weeks: known the previous day. Source: historical actuals.
- The same rates for route and hour, for slots with little history: known the previous day. Source: historical actuals.
- Holidays, school calendar, schedule-change weeks: known ahead of time. Source: calendar.
- Planned detours and service changes: known when announced. Source: service alerts.
- Weather forecast (optional): known the day before. Source: weather provider.
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
It starts in Task 2