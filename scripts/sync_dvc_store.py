"""Copy new DVC files into the public GitHub store, then push them with Git.

    python scripts/sync_dvc_store.py

Reviewers `dvc pull` from https://raw.githubusercontent.com/PrestonRamsay/pulse-dvc-store/main
(the default remote, no account needed). DVC can read that URL but not write to it, so new
files go through a local clone of the store:

    git clone https://github.com/PrestonRamsay/pulse-dvc-store.git ../pulse-dvc-store
    dvc remote add --local store ../pulse-dvc-store     # path stays in .dvc/config.local

This script runs `dvc push -r store` into that clone (DVC writes its files/md5/... layout),
commits whatever is new, and pushes it. It also pushes to DagsHub (`origin`) as a backup.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DVC = [sys.executable, "-m", "dvc"]


def run(cmd, cwd=ROOT):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def store_path():
    url = run([*DVC, "config", "--local", "remote.store.url"])
    path = (ROOT / ".dvc" / url).resolve() if not Path(url).is_absolute() else Path(url)
    if not (path / ".git").exists():
        raise SystemExit(f"{path} is not a clone of pulse-dvc-store (see this script's docstring)")
    return path


def main():
    store = store_path()
    run(["git", "pull", "--quiet"], cwd=store)
    print(run([*DVC, "push", "-r", "store"]))
    if not run(["git", "status", "--porcelain"], cwd=store):
        print("store already up to date")
    else:
        run(["git", "add", "-A"], cwd=store)
        run(["git", "commit", "--quiet", "-m", "Add DVC files"], cwd=store)
        run(["git", "push", "--quiet"], cwd=store)
        print("pushed new files to pulse-dvc-store")
    print(run([*DVC, "push", "-r", "origin"]))  # DagsHub backup (needs your login in config.local)


if __name__ == "__main__":
    main()
