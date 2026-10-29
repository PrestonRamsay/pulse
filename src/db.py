"""Settings shared by every script in src/: .env at the repo root fills in unset env vars."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5555/pulse"


def _load_dotenv(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def database_url():
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
