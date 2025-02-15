import math
import os
import random
from collections.abc import Callable
from pathlib import Path

import category_encoders as ce
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import torch

# from schedulefree import AdamWScheduleFree
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# from tqdm import tqdm
from .tabm_reference import Model, make_parameter_groups


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# https://www.kaggle.com/datasets/jsday96/mcts-tabm-models/data?select=TabMRegressor.py
class TabMRegressor:  # base class
    def __init__(
        self,
        arch_type: str = "tabm-mini",
        backbone: dict | None = None,
        d_embedding: int = 64,  # Only used for 'tabm-mini'
        bin_count: int = 52,  # Only used for 'tabm-mini'
        k: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-3,
        clip_grad_norm: bool = True,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 32,
        compile_model: bool = False,
        device: str | None = "cuda:0",
        random_state: int | None = 42,
        verbose: bool = True,
        categorical_features: list[str] | None = None,
        eval_metric: Callable | None = None,
        loss_fn: torch.nn.Module | None = None,
        checkpoint_path: str | Path | None = Path("best_model.pth"),
        scaling_method: str = "standard",
        reduce_lr_patience: int = 5,
    ):
        self.arch_type = arch_type
        self.backbone = backbone or {"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.1}
        self.d_embedding = d_embedding
        self.bin_count = bin_count
        self.k = k
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = verbose
        self.categorical_features = categorical_features
        self.eval_metric = eval_metric or mean_squared_error
        self.loss_fn = loss_fn
        self.checkpoint_path = Path(checkpoint_path)
        self.reduce_lr_patience = reduce_lr_patience

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.scaling_method = scaling_method
        seed_everything(random_state)

    def fit(  # noqa: C901
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray],
        sample_weight: np.ndarray | None = None,
    ):
        # PREPROCESS DATA.
        X_cat_train, X_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)

        # sample_weight が指定されていればテンソルに変換
        if sample_weight is not None:
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)

        # CREATE MODEL & TRAINING ALGO.
        bins = (
            rtdl_num_embeddings.compute_bins(X_cont_train, n_bins=self.bin_count)
            if self.arch_type == "tabm-mini"
            else None
        )
        self.model = Model(
            n_num_features=X_cont_train.shape[1],
            cat_cardinalities=cat_cardinalities,
            n_classes=None,
            backbone=self.backbone,
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    "type": "PiecewiseLinearEmbeddings",
                    "d_embedding": self.d_embedding,
                    "activation": False,
                    "version": "B",
                }
            ),
            arch_type=self.arch_type,
            k=self.k,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            make_parameter_groups(self.model),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=self.reduce_lr_patience,
            verbose=True,
            min_lr=1e-10,
        )
        if self.compile_model:
            self.model = torch.compile(self.model)

        try:
            loss_fn = self.loss_fn.to(self.device)
        except Exception:
            loss_fn = self.loss_fn

        # TRAIN & TEST MODEL.
        best = {
            "epoch": -1,
            "eval_loss": math.inf,
            "model_state_dict": None,
            "eval_score": -math.inf,
        }
        remaining_patience = self.patience
        header_printed = False
        for epoch in range(self.max_epochs):
            # TRAIN.
            optimizer.zero_grad()
            train_losses = []
            progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
            for batch_idx in progress_bar:
                self.model.train()

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y_pred = (
                        self.model(
                            X_cont_train[batch_idx],
                            X_cat_train[batch_idx],
                        )
                        .squeeze(-1)
                        .float()
                    )

                # ターゲットは各サンプルごとに k 回繰り返す
                y_target = y_train[batch_idx].repeat_interleave(self.k)
                y_pred_flat = y_pred.flatten(0, 1)

                # sample_weight が指定されている場合は、各サンプルに重みを適用
                if sample_weight is not None:
                    # 現在のバッチに対応する重みを k 回繰り返す
                    batch_weights = sample_weight[batch_idx].repeat_interleave(self.k)
                    # loss_fn が reduction 引数をサポートしている前提で、個別の損失値を計算
                    loss_values = loss_fn(y_pred_flat, y_target, reduction="none")
                    loss = (loss_values * batch_weights).mean()
                else:
                    loss = loss_fn(y_pred_flat, y_target)

                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # EVALUATE.
            val_y_preds = self.predict(eval_set[0], batch_size=self.batch_size)
            val_y_targets = eval_set[1]

            # PRINT INFO.
            mean_train_loss = np.mean(train_losses)
            val_score = self.eval_metric(y_true=val_y_targets, y_pred=val_y_preds)
            val_loss = loss_fn(torch.tensor(val_y_preds), torch.tensor(val_y_targets)).item()
            scheduler.step(val_score)  # update learning rate

            update_best = val_score > best["eval_score"]
            if update_best:
                best_score = val_score
                best_epoch = epoch
            else:
                best_score = best["eval_score"]
                best_epoch = best["epoch"]

            if self.verbose:
                if not header_printed:
                    print(
                        f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Score':<10} | {'Best Score':<10} | {'Best Epoch':<10} |"
                    )
                    print("-" * 90)
                    header_printed = True

                print(
                    f"{epoch:<10} | {mean_train_loss:<12.6f} | {val_loss:<10.6f} | {val_score:<10.6f} | {best_score:<10.6f} | {best_epoch:<10} |"
                )

            # COMPARE TO BEST.
            if update_best:
                best["epoch"] = epoch
                best["eval_loss"] = val_loss
                best["eval_score"] = val_score
                # save to checkpoint
                torch.save(obj=self.model.state_dict(), f=self.checkpoint_path)
                remaining_patience = self.patience
            else:
                remaining_patience -= 1

            # EARLY STOPPING.
            if remaining_patience == 0:
                break

        # RESTORE BEST MODEL.
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        best_pred = self.predict(eval_set[0], batch_size=self.batch_size)
        best_score = self.eval_metric(y_true=eval_set[1], y_pred=best_pred)
        print(f"Best score: {best_score}")

    def predict(self, X: pd.DataFrame, batch_size: int | None = 8096) -> np.ndarray:
        # PREPROCESS DATA.
        X_cat, X_cont, _, _ = self._preprocess_data(X, y=None, training=False)

        # PREDICT.
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_idx in torch.arange(0, len(X), batch_size, device=self.device):
                y_pred.append(
                    self.model(
                        X_cont[batch_idx : batch_idx + batch_size],
                        X_cat[batch_idx : batch_idx + batch_size],
                    )
                    .squeeze(-1)
                    .float()
                    .cpu()
                    .numpy()
                )

        y_pred = np.concatenate(y_pred)

        # COMPUTE ENSEMBLE MEAN.
        y_pred = np.mean(y_pred, axis=1)
        return y_pred

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, training: bool):
        # PICK NON-CONSTANT COLUMNS.
        if training:
            self._non_constant_columns = X.columns[X.nunique() > 1]

        X = X[self._non_constant_columns]

        # SEPARATE CATEGORICAL & CONTINUOUS FEATURES.
        X_cat = X[self.categorical_features].astype(str)
        X_cont = X.drop(columns=self.categorical_features).to_numpy()

        if training:
            self._categorical_encoders = ce.OrdinalEncoder().fit(X_cat)

        # ENCODE CATEGORICAL FEATURES.
        X_cat = self._categorical_encoders.transform(X_cat).to_numpy()
        X_cat[X_cat < 0] = 0  # -1,-2 -> 0
        cat_cardinalities = [X_cat[:, i].max() for i in range(X_cat.shape[1])]

        # SCALE CONTINUOUS FEATURES.
        if training:
            if self.scaling_method == "standard":
                self._cont_feature_preprocessor = preprocessing.StandardScaler().fit(X_cont)
            elif self.scaling_method == "minmax":
                self._cont_feature_preprocessor = preprocessing.MinMaxScaler().fit(X_cont)
            elif self.scaling_method == "robust":
                self._cont_feature_preprocessor = preprocessing.RobustScaler().fit(X_cont)
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        X_cont = self._cont_feature_preprocessor.transform(X_cont)

        # CONVERT TO TENSORS.
        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)

        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        return X_cat, X_cont, cat_cardinalities, y
