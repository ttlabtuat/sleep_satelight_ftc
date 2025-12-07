# train_2_pt.py
import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.model_selection import KFold
import os
import yaml
from ssft_pytorch import build_model_pt
from dataset_pt import SleepDataset
from dataset_utils import get_dataset_params

class Config:
    def __init__( self, train_cfg, dataset_cfg, model_cfg ):
        self.dataset = dataset_cfg["name"]
        self.dataset_dir = dataset_cfg["dir"]
        self.dataset_fs = dataset_cfg["fs"]

        self.base_model_dir = train_cfg["base_model_dir"]
        self.save_model_dir = train_cfg["save_model_dir"]
        self._ensure_directory(self.save_model_dir)

        self._set_dataset_params()

        self.seed = train_cfg.get("seed", 0)

        self.lr = train_cfg.get("lr", 0.001)
        self.batch_size   = train_cfg.get("batch_size", 64)
        self.epochs       = train_cfg.get("epochs", 50)
        self.patience     = train_cfg.get("patience", 10)

        self.model_params = model_cfg

        # ---- transfer learning params ----
        self.context_length   = train_cfg.get("context_length", 15)
        self.feature_dim      = train_cfg.get("feature_dim", 300)
        self.transfer_hidden  = train_cfg.get("transfer_hidden", 128)
        self.num_classes      = model_cfg.get("num_classes", 5)

        self.transfer_lr      = train_cfg.get("transfer_lr", 0.001)
        self.transfer_epochs  = train_cfg.get("transfer_epochs", 50)
        self.transfer_patience= train_cfg.get("transfer_patience", 10)

    def _ensure_directory(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

    def _set_dataset_params(self) -> None:
        params = get_dataset_params(self.dataset, self.dataset_fs)
        self.dataset_ids = params["ids"]
        self.dataset_num = params["num"]
        self.fold_num = params["fold_num"]
        self.assignments = params["assignments"]


def load_feature_model(checkpoint, model_params):
    model = build_model_pt(**model_params)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model.cuda()

def extract_all_features(dataset, base_model):
    """
    Iterate dataset → get feature per sample → concat.
    dataset returns (xt, xf, label)
    base_model.extract_feature returns (1, feature_dim)
    """
    feat_list = []
    label_list = []

    for xt, xf, y in dataset:
        # shape (C,L,1) → (1,C,L,1)
        ft = base_model.extract_feature(
            xt.unsqueeze(0).cuda(),
            xf.unsqueeze(0).cuda()
        ).cpu()  # (1,feature_dim)
        feat_list.append(ft)
        label_list.append(y)

    # (N,feature_dim)
    feat = torch.cat(feat_list, dim=0)
    # (N,)
    labels = torch.stack(label_list, dim=0)

    return feat, labels

def make_context_samples_torch(features, labels, context_len):
    N, F = features.shape
    M = N - context_len + 1
    center = context_len // 2

    X = torch.zeros((M, context_len, F), dtype=torch.float32)
    y = torch.zeros((M,), dtype=torch.long)

    for i in range(M):
        X[i] = features[i:i+context_len]
        y[i] = labels[i+center]

    return X, y


class TransferMLP(nn.Module):
    def __init__(self, feature_dim=300, context_len=15, hidden_dim=128, num_classes=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_dim * context_len, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def run_transfer_training(config):
    kf = KFold(n_splits=config.fold_num, shuffle=True, random_state=config.seed)

    for fold_count, (train_subs, val_subs) in enumerate(
        kf.split(range(config.fold_num))
    ):
        save_fold_dir = os.path.join(config.save_model_dir, f"fold{fold_count+1:02d}")
        os.makedirs(save_fold_dir, exist_ok=True)

        base_ckpt = os.path.join(config.base_model_dir, f"fold{fold_count+1:02d}_best.pt")
        base_model = load_feature_model(base_ckpt, config.model_params)

        train_ids = np.concatenate([config.assignments[i] for i in train_subs])
        val_ids   = np.concatenate([config.assignments[i] for i in val_subs])

        train_ds = SleepDataset(config.dataset_dir, train_ids, config.dataset_fs)
        val_ds   = SleepDataset(config.dataset_dir, val_ids, config.dataset_fs)

        train_feat, train_label = extract_all_features(train_ds, base_model)
        val_feat,   val_label   = extract_all_features(val_ds, base_model)

        Xtrain, Ytrain = make_context_samples_torch(train_feat, train_label, config.context_length)
        Xval,   Yval   = make_context_samples_torch(val_feat,   val_label,   config.context_length)

        model = TransferMLP(
            feature_dim=config.feature_dim,
            context_len=config.context_length,
            hidden_dim=config.transfer_hidden,
            num_classes=config.num_classes
        ).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.transfer_lr)

        history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")
        best_epoch = -1
        best_val_acc = 0.0
        best_train_acc = 0.0
        patience_cnt = 0

        for epoch in range(config.transfer_epochs):
            model.train()
            optimizer.zero_grad()
            Yhat = model(Xtrain.cuda())
            loss = criterion(Yhat, Ytrain.cuda())
            loss.backward()
            optimizer.step()

            # Acc
            _, pred = torch.max(Yhat, 1)
            acc = (pred.cpu() == Ytrain).float().mean().item()

            # validation
            model.eval()
            with torch.no_grad():
                Yhat_val = model(Xval.cuda())
                val_loss = criterion(Yhat_val, Yval.cuda())
                _, predv = torch.max(Yhat_val, 1)
                val_acc = (predv.cpu() == Yval).float().mean().item()
    
            val_loss_value = val_loss.item()

            print(f"[Fold {fold_count+1:02d}] Epoch {epoch+1}/{config.transfer_epochs} "
                  f"loss={loss:.3f}, acc={acc:.3f}, "
                  f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")

            history["loss"].append(loss.item())
            history["accuracy"].append(acc)
            history["val_loss"].append(val_loss_value)
            history["val_accuracy"].append(val_acc)

            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_epoch = epoch
                best_val_acc = val_acc
                best_train_acc = acc
                patience_cnt = 0

                torch.save(model.state_dict(),
                    os.path.join(save_fold_dir, f"fold{fold_count+1:02d}_best_transfer.pt"))
            else:
                patience_cnt += 1
                if patience_cnt >= config.transfer_patience:
                    break

        history["best_val_loss"] = best_val_loss
        history["best_epoch"] = best_epoch
        history["best_val_acc"] = best_val_acc
        history["best_train_acc"] = best_train_acc
        
        history_path = os.path.join(save_fold_dir, f"history-fold{fold_count+1:02d}.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        print(f"Saved {history_path}")

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get("dataset", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    train2_cfg = cfg.get("train_2", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    train2_params = {
        **train_cfg,
        **train2_cfg,
    }

    config = Config(
       train_cfg=train2_params,
       dataset_cfg=dataset_cfg,
       model_cfg=model_cfg
    )

    run_transfer_training(config)
