# train_1_pt.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import yaml
import pickle
import os
from ssft_pytorch import build_model_pt
from dataset_pt import SleepDataset
from dataset_utils import get_dataset_params

class Config:
    """
    Configuration holder loaded from YAML.
    """
    def __init__( self, train_cfg, dataset_cfg, model_cfg ):
        self.dataset = dataset_cfg["name"]
        self.dataset_dir = dataset_cfg["dir"]
        self.dataset_fs = dataset_cfg["fs"]

        self.save_model_dir = train_cfg["save_model_dir"]
        self._ensure_directory(self.save_model_dir)

        self._set_dataset_params()

        self.seed = train_cfg.get("seed", 0)
        self.fs = train_cfg.get("fs", 100)
        self.epochs = train_cfg.get("epochs", 100)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.patience = train_cfg.get("patience", 10)

        self.class_weight = train_cfg.get("class_weight", None)
        self.lr = train_cfg.get("lr", 0.001)

        self.model_params = model_cfg

    def _ensure_directory(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

    def _set_dataset_params(self) -> None:
        params = get_dataset_params(self.dataset, self.dataset_fs)
        self.dataset_ids = params["ids"]
        self.dataset_num = params["num"]
        self.fold_num = params["fold_num"]
        self.assignments = params["assignments"]


def run_training_pt(config):
    kf = KFold(n_splits=config.fold_num, shuffle=True, random_state=config.seed)

    for fold_count, (train_subs, val_subs) in enumerate(
        kf.split(range(config.fold_num))
    ):
        train_ids = np.concatenate([config.assignments[i] for i in train_subs])
        val_ids   = np.concatenate([config.assignments[i] for i in val_subs])

        # Dataset
        train_ds = SleepDataset(config.dataset_dir, train_ids, config.dataset_fs)
        val_ds   = SleepDataset(config.dataset_dir, val_ids, config.dataset_fs)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False)

        # Model
        model = build_model_pt(**config.model_params).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        best_val_loss = float("inf")
        best_epoch = -1
        bast_model_state = None
        patience_counter = 0

        history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.0
            correct = 0

            for xt, xf, y in train_loader:
                xt, xf, y = xt.cuda(), xf.cuda(), y.cuda()

                optimizer.zero_grad()
                y_hat = model(xt, xf)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                correct += (y_hat.argmax(1) == y).sum().item()

            avg_loss = total_loss / len(train_ds)
            acc = correct / len(train_ds)

            # validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for xt, xf, y in val_loader:
                    xt, xf, y = xt.cuda(), xf.cuda(), y.cuda()
                    y_hat = model(xt, xf)
                    loss = criterion(y_hat, y)

                    val_loss += loss.item() * y.size(0)
                    val_correct += (y_hat.argmax(1) == y).sum().item()

            avg_val_loss = val_loss / len(val_ds)
            val_acc = val_correct / len(val_ds)

            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_acc)

            print(f"Epoch {epoch+1}/{config.epochs} "
                  f"loss={avg_loss:.3f}, acc={acc:.3f}, "
                  f"val_loss={avg_val_loss:.3f}, val_acc={val_acc:.3f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping")
                    break

        # save model
        model_path = os.path.join(
            config.save_model_dir,
            f"fold{fold_count+1:02d}_best.pt",
        )
        if best_model_state is not None:
            torch.save(best_model_state, model_path)
        else:
            torch.save(model.state_dict(), model_path)
        
        history["best_val_loss"] = best_val_loss
        history["best_epoch"] = best_epoch
        
        history_path = os.path.join(
            config.save_model_dir,
            f"history-fold{fold_count+1:02d}.pkl",
        )
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        print(f"Saved {model_path}")
        print(f"Saved {history_path}")

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get("dataset", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    train1_cfg = cfg.get("train_1", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    train1_params = {
        **train_cfg,
        **train1_cfg,
    }

    config = Config(
        train_cfg=train1_params,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
    )


    run_training_pt(config)