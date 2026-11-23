"""Train the latenet neural network from a YAML config, logging every epoch to MLflow.

    python -m src.train_latenet configs/latenet_base.yaml

Reads the frozen train and validation splits only; the test split is never read here.
"""

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

from src.latenet_data import build_scaling, build_vocab, load_train_val, make_loader
from src.latenet_model import build_model, count_parameters
from src.train import ARTIFACT_ROOT, TRACKING_URI, dvc_md5, export_runs, flatten, git_commit


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def train_one_epoch(model, loader, optimizer, loss_fn):
    """One pass over the train batches; returns the average loss per row."""
    model.train()
    total, rows = 0.0, 0
    for cat, num, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(cat, num), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        rows += len(y)
    return total / rows


@torch.no_grad()
def predict(model, loader):
    """Logits and labels for a whole split, in loader order."""
    model.eval()
    logits, labels = [], []
    for cat, num, y in loader:
        logits.append(model(cat, num))
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    t = cfg["train"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train, val, history_cols = load_train_val(cfg)
    vocab, scaling = build_vocab(train), build_scaling(train, history_cols)
    train_loader = make_loader(train, vocab, scaling, t["batch_size"], shuffle=True)
    val_loader = make_loader(val, vocab, scaling, t["batch_size"], shuffle=False)

    n_numeric = next(iter(val_loader))[1].shape[1]
    model = build_model(cfg, vocab, n_numeric)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        # Provenance: every config value, the code version and the data version.
        mlflow.log_params(flatten({k: v for k, v in cfg.items() if k not in ("experiment", "run_name")}))
        mlflow.log_params({
            "git_commit": git_commit(),
            "dvc_md5_train": dvc_md5("train"),
            "dvc_md5_val": dvc_md5("val"),
            "dvc_md5_test": dvc_md5("test"),
            "rows_train": len(train),
            "rows_val": len(val),
            "n_parameters": count_parameters(model),
            "n_numeric": n_numeric,
        })
        mlflow.log_artifact(str(args.config), artifact_path="config")
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.json"
            vocab_path.write_text(json.dumps({"vocab": vocab, "scaling": scaling}, indent=1))
            mlflow.log_artifact(str(vocab_path), artifact_path="vocab")

        best_auc, best_epoch, best_state, waited = -1.0, 0, None, 0
        for epoch in range(1, t["max_epochs"] + 1):
            started = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
            logits, y = predict(model, val_loader)
            proba = torch.sigmoid(logits).numpy()
            metrics = {
                "train_loss": train_loss,
                "val_loss": loss_fn(logits, y).item(),
                "val_auc_roc": roc_auc_score(y.numpy(), proba),
                "val_accuracy": accuracy_score(y.numpy(), proba >= t["threshold"]),
                "epoch_seconds": time.perf_counter() - started,
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            # Early stopping on validation AUC; keep a copy of the best weights.
            if metrics["val_auc_roc"] > best_auc + t["min_delta"]:
                best_auc, best_epoch, waited = metrics["val_auc_roc"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= t["patience"]:
                    print(f"stopping: no improvement for {t['patience']} epochs")
                    break

        model.load_state_dict(best_state)
        out = Path(cfg["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "vocab": vocab,
                    "scaling": scaling, "n_numeric": n_numeric,
                    "best_epoch": best_epoch, "best_val_auc_roc": best_auc}, out)
        print(f"saved {out}")
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_auc_roc": best_auc})
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        print(f"best epoch {best_epoch}: val_auc_roc={best_auc:.4f}")
    export_runs(cfg["experiment"])


if __name__ == "__main__":
    main()