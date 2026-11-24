# pulse

Predicts whether an MBTA bus will reach a timepoint stop more than 3 minutes late.
Spec: [ml_spec.md](ml_spec.md). Results: [REPORT.md](REPORT.md).

## Layout

```
src/        ingestion (load_historical, pull_daily, pull_lamp + lamp, lamp_check),
            preparation (make_splits, make_extra, extra), features,
            training (train_baseline for Task 2, train + models), calibrate, evaluate,
            the neural network (latenet_data, latenet_model, train_latenet, evaluate_latenet),
            latency_check
models/     DVC-tracked: pulse-latenet (the production model), latenet.pt (the best net)
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

## From a fresh clone to the leading run

After the setup above (the venv and `pip install -r requirements.txt`; no database or `.env`
is needed for this path):

```bash
dvc pull
python -m src.train configs/final.yaml
```

- `dvc pull` fetches the frozen train/val/test splits, the April–June file and the models
  (`models/pulse-latenet`, `models/latenet.pt`), about 82 MB. **No account or token is
  needed:** the default remote is the public GitHub store
  [`PrestonRamsay/pulse-dvc-store`](https://github.com/PrestonRamsay/pulse-dvc-store), read
  over HTTPS. A DagsHub copy (`origin`) is kept as a backup and needs a DagsHub login.
  Maintainers add new files with `python scripts/sync_dvc_store.py`, because DVC can't push
  to the GitHub store.
- The second command retrains the current leading run (`logreg_history_rates`, logistic regression on
  schedule fields + recent late rates) and logs it to a new local MLflow store. It prints
  validation **AUC 0.7255, accuracy 0.6622**, the same as the logged run, because the fit is
  deterministic. It takes about 2 minutes.

Optional follow-ups, each one command:

```bash
python -m src.calibrate logreg_history_rates                   # rolling recalibration + threshold table
python -m src.evaluate logreg_history_rates --threshold 0.40   # the single test evaluation (AUC 0.738)
python -m src.latency_check                                    # single-prediction CPU latency (~13 ms)
mlflow ui --backend-store-uri sqlite:///mlflow.db              # browse the runs ("Model training" view)
```

Results and the reasoning behind every choice: [REPORT.md](REPORT.md).

## From raw data to the baselines

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

   Then load yesterday's **actual** times from MBTA LAMP (run daily too). The API has no
   actual times; LAMP does, and it agrees with the monthly files on 99.92% of late labels.
   It keeps monthly-file rows untouched, and a monthly file replaces LAMP rows when it
   arrives. Counts are appended to `logs/pull_lamp.jsonl`.

   ```bash
   python -m src.pull_lamp                                       # last 7 finished days
   python -m src.pull_lamp --start 2026-10-01 --end 2026-11-17   # optional backfill
   python -m src.lamp_check                                      # LAMP vs monthly files, 1-15 Sep
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
   python -m src.train configs/final.yaml                   # the current leader (= logreg_history_rates)
   python -m src.calibrate logreg_history_rates             # rolling recalibration + threshold table
   ```

   The neural network (latenet) is trained per config with `src.train_latenet`. Its
   validation tables come from `src.evaluate_latenet`:

   ```bash
   python -m src.train_latenet configs/latenet_final.yaml   # best net, val AUC 0.716
   python -m src.evaluate_latenet val --threshold 0.35      # metrics, calibration by hour, routes, thresholds
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
