import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import torch

# from schedulefree import AdamWScheduleFree
from sklearn.preprocessing import MinMaxScaler

# from tqdm import tqdm
from .tabm_reference import Model, make_parameter_groups


# https://www.kaggle.com/datasets/jsday96/mcts-tabm-models/data?select=TabMRegressor.py
class TabMRegressor:
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
        random_state: int = 0,
        verbose: bool = True,
        categorical_features: list[str] | None = None,
        eval_metric: str | Callable = "mse",
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
        self.eval_metric = eval_metric

    def fit(self, X: pd.DataFrame, y: np.array, eval_set: tuple[pd.DataFrame, np.array]):  # noqa
        # PREPROCESS DATA.
        X_cat_train, X_cont_train, cat_cardinalities, y_train = self._preprocess_data(X, y, training=True)
        X_cat_val, X_cont_val, _, y_val = self._preprocess_data(eval_set[0], eval_set[1], training=False)

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
            make_parameter_groups(self.model), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.compile_model:
            self.model = torch.compile(self.model)

        loss_fn = torch.nn.MSELoss().to(self.device)

        # TRAIN & TEST MODEL.
        best = {
            "epoch": -1,
            "eval_loss": math.inf,
            "model_state_dict": None,
            "eval_score": -math.inf,
        }
        remaining_patience = self.patience
        # epoch_size = math.ceil(len(X) / self.batch_size)
        header_printed = False
        for epoch in range(self.max_epochs):
            # TRAIN.
            optimizer.zero_grad()
            train_losses = []
            progress_bar = torch.randperm(len(y_train), device=self.device).split(self.batch_size)
            # progress_bar = tqdm(progress_bar, desc=f"Epoch {epoch}", total=epoch_size) if self.verbose else progress_bar
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

                loss = loss_fn(y_pred.flatten(0, 1), y_train[batch_idx].repeat_interleave(self.k))
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # EVALUATE.
            self.model.eval()
            val_losses = []
            val_y_preds = []
            val_y_targets = []
            with torch.no_grad():
                for batch_idx in torch.arange(0, len(y_val), self.batch_size, device=self.device):
                    y_pred = (
                        self.model(
                            X_cont_val[batch_idx : batch_idx + self.batch_size],
                            X_cat_val[batch_idx : batch_idx + self.batch_size],
                        )
                        .squeeze(-1)
                        .float()
                    )

                    loss = loss_fn(
                        y_pred.flatten(0, 1),
                        y_val[batch_idx : batch_idx + self.batch_size].repeat_interleave(self.k),
                    )
                    val_losses.append(loss.item())
                    val_y_preds.append(y_pred.cpu().numpy())
                    val_y_targets.append(y_val[batch_idx : batch_idx + self.batch_size].cpu().numpy())

            # PRINT INFO.
            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)

            # NOTE: prod std and plus mean?
            # (n_samples, k) -> (n_samples,)
            val_y_preds = np.concatenate(val_y_preds).mean(axis=1) * self._target_std + self._target_mean
            val_y_targets = np.concatenate(val_y_targets)

            if self.eval_metric == "mse":
                val_score = -mean_val_loss
            else:
                val_score = self.eval_metric(y_true=val_y_targets, y_pred=val_y_preds)

            update_best = val_score > best["eval_score"]
            if update_best:
                best_score = val_score
            else:
                best_score = best["eval_score"]

            if self.verbose:
                if not header_printed:
                    print(
                        f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Score':<10} | {'Best Score':<10}"
                    )
                    print("-" * 60)
                    header_printed = True

                print(
                    f"{epoch:<10} | {mean_train_loss:<12.6f} | {mean_val_loss:<10.6f} | {val_score:<10.6f} | {best_score:<10.6f}"
                )

            # COMPARE TO BEST.
            if update_best:
                best["epoch"] = epoch
                best["eval_loss"] = mean_val_loss
                best["eval_score"] = val_score
                best["model_state_dict"] = self.model.state_dict()
                remaining_patience = self.patience

                # if self.verbose:
                #     print(f"🎉 New best model found! | Epoch: {epoch} | Best Score: {val_score}")
            else:
                remaining_patience -= 1

            # EARLY STOPPING.
            if remaining_patience == 0:
                break

        # RESTORE BEST MODEL.
        self.model.load_state_dict(best["model_state_dict"])

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

        # DENORMALIZE TARGETS.
        y_pred = y_pred * self._target_std + self._target_mean

        # COMPUTE ENSEMBLE MEAN.
        y_pred = np.mean(y_pred, axis=1)

        return y_pred

    def _preprocess_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        training: bool,
    ):
        # PICK NON-CONSTANT COLUMNS.
        if training:
            self._non_constant_columns = X.columns[X.nunique() > 1]

        X = X[self._non_constant_columns]

        # SEPARATE CATEGORICAL & CONTINUOUS FEATURES.
        X_cat = X[self.categorical_features].to_numpy()
        X_cont = X.drop(columns=self.categorical_features).to_numpy()

        # ENCODE CATEGORICAL FEATURES.
        cat_cardinalities = [X[col].nunique() for col in self.categorical_features]

        # NORMALIZE TARGETS.
        if training:
            self._target_mean = y.mean()
            self._target_std = y.std()

            y = (y - self._target_mean) / self._target_std

        # SCALE CONTINUOUS FEATURES.
        if training:
            noise = np.random.default_rng(0).normal(0.0, 1e-5, X_cont.shape).astype(X_cont.dtype)
            self._cont_feature_preprocessor = MinMaxScaler().fit(X_cont + noise)

        X_cont = self._cont_feature_preprocessor.transform(X_cont)

        # CONVERT TO TENSORS.
        X_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device)

        if y is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        return X_cat, X_cont, cat_cardinalities, y
