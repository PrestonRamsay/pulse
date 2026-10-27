# pulse

Predicts whether an MBTA bus will reach a stop more than 3 minutes late.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
docker compose up -d             # Postgres on localhost:5432; creates stop_events from src/schema.sql
```

The default connection is `postgresql://pulse:pulse@localhost:5432/pulse`. Set `DATABASE_URL` to use another database (see `.env.example`).

## Data

Download the 2026 MBTA Bus Arrival/Departure zip into `data/raw/` and extract it, keeping the three most recent consecutive months:

```bash
curl -L -o data/raw/mbta_2026.zip "https://www.arcgis.com/sharing/rest/content/items/9d8a8cad277545c984c1b25ed10b7d3c/data"
python -c "import zipfile; zipfile.ZipFile('data/raw/mbta_2026.zip').extractall('data/raw')"
```

Load them into `stop_events` (safe to rerun):

```bash
python -m src.load_historical
```

## Tests

```bash
pytest
```
