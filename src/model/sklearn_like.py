import copy
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import xgboost as xgb
from lightgbm import LGBMModel
from numpy.typing import NDArray
from xgboost import XGBRegressor


class BaseWrapper:
    model: Any
    name: str

    def fit(self, tr_x: NDArray, tr_y: NDArray, va_x: NDArray, va_y: NDArray) -> None:
        raise NotImplementedError

    def predict(self, X: NDArray) -> NDArray:  # noqa
        raise NotImplementedError

    def save(self, out_dir: str | Path) -> None:
        path = self.get_save_path(out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, out_dir: Path | str) -> None:
        path = self.get_save_path(out_dir)
        self.model = joblib.load(path)
        self.fitted = True

    def get_save_path(self, out_dir: Path | str) -> Path:
        return Path(out_dir) / "model.pkl"

    def is_saved(self, out_dir: Path | str) -> bool:
        path = self.get_save_path(out_dir)
        return path.exists()


class LightGBMWapper(BaseWrapper):
    def __init__(
        self,
        name: str = "lgb",
        model: LGBMModel | None = None,
        fit_params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.name = name
        self.model = model or LGBMModel()
        self.fit_params = fit_params or {}
        self.fitted = False
        self.feature_names = feature_names or self.fit_params.get("feature_name")

        self.params = self.model.get_params()
        self.eval_metric = self.fit_params.get("eval_metric")

    def initialize(self) -> None:
        params = copy.deepcopy(self.params)
        self.model = LGBMModel(**params)

    def reshape_y(self, y: NDArray) -> NDArray:
        if y.ndim == 1:
            return y
        if y.shape[1] == 1:
            return y.reshape(-1)
        return y

    def fit(self, tr_x: NDArray, tr_y: NDArray, va_x: NDArray, va_y: NDArray) -> None:
        self.initialize()
        self.model.fit(
            tr_x,
            self.reshape_y(tr_y),
            eval_set=[(va_x, self.reshape_y(va_y))],
            **self.fit_params,
        )
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if not self.fitted:
            raise ValueError("Model is not fitted yet")
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> Any:
        return self.model.feature_importances_


class XGBoostRegressorWrapper(BaseWrapper):
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    def __init__(
        self,
        name: str = "xgb",
        model: XGBRegressor | None = None,
        fit_params: dict[str, Any] | None = None,
        early_stopping_params: dict[str, Any] | None = None,  # callbacks を引き継いでしまうので注意
        cat_features: list[int] | None = None,
        feature_names: list[str] | None = None,
    ):
        self.name = name
        self.model = model or XGBRegressor()
        self.fit_params = fit_params or {}
        self.fitted = False
        self.early_stopping_params = early_stopping_params or {}
        self.cat_features = cat_features
        self.feature_names = feature_names

        self.params = self.model.get_params()
        self.eval_metric = self.params.get("eval_metric")

    def initialize(self) -> None:
        params = copy.deepcopy(self.params)
        params["eval_metric"] = self.eval_metric

        if self.early_stopping_params:
            # 同じ instance だとバグるので initialize する
            callbacks = params.get("callbacks") or []
            early_stopping_callback = xgb.callback.EarlyStopping(**self.early_stopping_params)
            params["callbacks"] = callbacks + [early_stopping_callback]

        self.model = XGBRegressor(**params)

    def fit(self, tr_x: NDArray, tr_y: NDArray, va_x: NDArray, va_y: NDArray) -> None:
        self.initialize()

        if self.cat_features:
            tr_x = (
                pl.DataFrame(tr_x, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )
            va_x = (
                pl.DataFrame(va_x, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )
        self.model.fit(
            tr_x,
            tr_y,
            eval_set=[(va_x, va_y)],
            **self.fit_params,
        )
        self.fitted = True

    def predict(self, X: NDArray) -> NDArray:  # noqa
        if not self.fitted:
            raise ValueError("Model is not fitted yet")

        if self.cat_features:
            X = (
                pl.DataFrame(X, schema=self.feature_names)
                .cast({x: pl.String for x in self.cat_features})
                .cast({x: pl.Categorical for x in self.cat_features})
                .to_pandas()
            )
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> Any:
        return self.model.feature_importances_
