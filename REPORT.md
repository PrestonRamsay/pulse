# Pulse Model Delivery Report

## 1. Data

## Data

### Dataset and coverage

The historical dataset contains nine consecutive monthly MBTA Bus Arrival/Departure CSV files covering January through September 2026. The files contain **22,177,874 rows** in total, with monthly row counts ranging from 2,197,101 in February to 2,543,650 in May.

The dataset includes service date, route, direction, stop, time-point metadata, scheduled timestamps, actual timestamps, and headway-related fields. The observed record types are `Headway` and `Schedule`, and the observed point types are `Startpoint`, `Midpoint`, and `Endpoint`.

### Data quality

No scheduled timestamps were missing, and no invalid nonblank timestamps were detected by the initial inspection script. However, 1,147,354 rows (5.2%) had missing actual timestamps. These rows cannot be assigned a historical delay label from the available timestamps.

The `headway` column was missing in 2,482,644 rows (11.2%), while `scheduled_headway` was missing in 584,832 rows (2.6%). The `half_trip_id` column had only 550 missing values.

The initial uniqueness check found 71,147 excess occurrences of the proposed key `(service_date, route_id, direction_id, stop_id, scheduled)`. These records require further investigation to determine whether they represent exact duplicates or distinct observations sharing the same key.

### Delay distribution

For rows with both scheduled and actual timestamps, delay was calculated as actual time minus scheduled time in minutes. The initial analysis identified 1,900 delays below −30 minutes and 274 delays above 120 minutes. After excluding these outliers, the retained delay distribution had a median of 2.42 minutes and a 90th percentile of 10.27 minutes.

Using the project's threshold of more than three minutes late, 44.2% of the retained timestamp-pair observations were late. This is a preliminary estimate; the final class balance will be recalculated after applying the complete training-data filters.

### Initial cleaning considerations

The scheduled and actual timestamps use a `1900-01-01` date component in the inspected records. The time-of-day values must be combined with `service_date` according to the project's wall-clock convention. Route IDs must also preserve their historical string representation during ingestion because some numeric route IDs are zero-padded and others are special identifiers such as `SL1`.

Before finalizing the training dataset, record-type-specific missingness and the duplicate-key cases will be investigated. Rows without actual timestamps will not be used as labeled historical training examples.

### Duplicate-key investigation

A follow-up analysis grouped records by the proposed key `(service_date, route_id, direction_id, stop_id, scheduled)`. It found 71,036 keys occurring more than once, representing 71,147 excess rows. Of these repeated keys, 71,001 had differences in at least one of the compared fields, including `actual`, `standard_type`, `point_type`, `half_trip_id`, `scheduled_headway`, or `headway`. Only 35 excess rows repeated the same key and the same values across those compared fields.

Examples showed that records sharing the proposed key could have different `half_trip_id` values and, in some cases, different actual timestamps. Therefore, repeated keys cannot automatically be treated as duplicate records.

The proposed key's uniqueness assumption requires further consideration. The project specifies this key for database conflict handling, so any decision to use a different key or exclude conflicting records must be documented and reconciled with that requirement.


## 2. Baselines
- Majority-class: accuracy
- Logistic regression: accuracy, AUC, PR-AUC
- What the baselines tell us about the problem

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