# DelayNet
Predicts whether MBTA bus trips run >3 min late.
Pipeline: MBTA API -> Postgres -> training. Spec in ml_spec.md.

## Commands
- Activate env: venv\Scripts\activate (Windows; Mac/Linux: source venv/bin/activate)
- Run tests: pytest
- Run ingestion once: python -m src.ingest

## Rules
- "Late" = strictly more than 3 minutes after scheduled time.
- Features must be knowable BEFORE the trip runs. No post-trip data. Ever.
- Data logic in src/. Notebooks explore; they don't define the pipeline.
- Never modify data/raw/.
