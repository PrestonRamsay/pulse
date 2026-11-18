# pulse

Predicts whether an MBTA bus will reach a timepoint stop more than 3 minutes late.
Spec: [ml_spec.md](ml_spec.md). Results: [REPORT.md](REPORT.md).

## Layout

```
src/        ingestion (load_historical, pull_daily), preparation (make_splits), features,
            training (train_baseline for Task 2, train + models), calibrate, evaluate
configs/    one YAML per training run
data/       DVC-tracked; only the .dvc pointer files are in Git
notebooks/  exploration only
tests/      pytest
assets/     images referenced from REPORT.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the repo root (it's gitignored) with your database and, optionally,
a free MBTA API key from [api-v3.mbta.com](https://api-v3.mbta.com):

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5555/pulse
MBTA_API_KEY=your_key
```

`docker compose up -d` also gives you a database, with
`DATABASE_URL=postgresql://pulse:pulse@localhost:5432/pulse`. The scripts create the
`stop_events` table from `src/schema.sql` if it doesn't exist.

## From a fresh clone to the baselines

1. Get the raw files (one zip per year; one CSV per month):

   ```bash
   curl -L -o data/raw/mbta_2026.zip "https://www.arcgis.com/sharing/rest/content/items/9d8a8cad277545c984c1b25ed10b7d3c/data"
   python -c "import zipfile; zipfile.ZipFile('data/raw/mbta_2026.zip').extractall('data/raw')"
   ```

2. Load the three most recent months into `stop_events` (safe to rerun):

   ```bash
   python -m src.load_historical
   ```

3. Pull today's planned service from the MBTA API into the same table (run daily; set
   `MBTA_API_KEY` in `.env` to lift the 20 requests/minute limit). Counts are appended to
   `logs/pull_daily.log`.

   ```bash
   python -m src.pull_daily
   ```

4. Build the time-based splits, or fetch the frozen ones with `dvc pull` if you have the remote:

   ```bash
   python -m src.make_splits
   dvc add data/splits/train.parquet data/splits/val.parquet data/splits/test.parquet
   ```

5. Train and score the baselines on validation (logged to `mlflow.db` / `mlruns/`; run
   table exported to `mlflow_runs.csv`):

   ```bash
   python -m src.train_baseline configs/baseline_majority.yaml
   python -m src.train_baseline configs/baseline_logreg.yaml
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

6. Experiments (Task 3). Each config is one MLflow run, scored on validation, with
   validation predictions saved to `data/preds/`. History features use only earlier
   service days.

   ```bash
   python -m src.train configs/history_slot_8wk.yaml        # spec baseline, no training
   python -m src.train configs/gbt_history.yaml             # gradient-boosted trees
   python -m src.train configs/logreg_history_rates.yaml    # the chosen model
   python -m src.calibrate logreg_history_rates             # rolling recalibration + threshold table
   ```

7. Final evaluation, run once. This is the only step that reads the test split. It
   registers `pulse-latenet` and promotes it to Production if it beats every baseline:

   ```bash
   python -m src.evaluate logreg_history_rates --threshold 0.40 --register
   ```

## Tests

```bash
pytest
```
