import numpy as np
import polars as pl
from lifelines.utils import concordance_index


class Metric:
    # https://www.kaggle.com/code/metric/eefs-concordance-index
    def __init__(
        self,
        event_label: str = "efs",
        interval_label: str = "efs_time",
        prediction_label: str = "pred",
        group_label: str = "race_group",
        name: str | None = None,
    ):
        self.event_label = event_label
        self.interval_label = interval_label
        self.prediction_label = prediction_label
        self.group_label = group_label
        self._name = name or self.__class__.__name__

    def __call__(self, input_df: pl.DataFrame) -> float:
        y_time = input_df[self.interval_label].to_numpy()
        y_event = input_df[self.event_label].to_numpy()
        y_pred = input_df[self.prediction_label].to_numpy()
        race_group = input_df[self.group_label].to_numpy()
        score = metric(y_time=y_time, y_event=y_event, y_pred=y_pred, race_group=race_group)
        return score

    @property
    def __name__(self) -> str:
        return self._name


def metric(
    y_time: np.ndarray,
    y_event: np.ndarray,
    y_pred: np.ndarray,
    race_group: np.ndarray,
) -> float:
    """
    グループごとのC-indexの(平均 - sqrt(分散))を計算する関数 (df 使わないバージョン)

    Parameters
    ----------
    y_time : np.ndarray
        イベントまたはセンサーされた時間(efs_timeに相当)
    y_event : np.ndarray
        イベントが発生したかどうかのフラグ(efsに相当, 1 or 0)
    y_pred : np.ndarray
        モデルからの予測値(predに相当)
    race_group : np.ndarray
        サンプルごとのグループを表す配列(race_groupに相当)

    Returns
    -------
    float
        グループごとのC-indexの(平均 - sqrt(分散))
    """

    unique_groups = np.unique(race_group)
    metric_list = []

    for g in unique_groups:
        indices = np.where(race_group == g)
        # Concordance index計算
        c_index_race = concordance_index(
            y_time[indices],
            -y_pred[indices],
            y_event[indices],
        )
        metric_list.append(c_index_race)

    # 平均 - sqrt(分散)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))


class XGBMetric:
    def __init__(
        self,
        event_label: str = "efs",
        interval_label: str = "efs_time",
        group_label: str = "race_group",
        name: str | None = None,
    ):
        self.va_df: pl.DataFrame | None = None
        self.event_label = event_label
        self.interval_label = interval_label
        self.group_label = group_label
        self._name = name or self.__class__.__name__

    def __call__(self, y_true, y_pred) -> float:
        assert self.va_df is not None, "va_df is not set"

        if len(y_true) != len(self.va_df):
            # training phase
            return 0.0

        y_time = self.va_df[self.interval_label].to_numpy()
        y_event = self.va_df[self.event_label].to_numpy()
        race_group = self.va_df[self.group_label].to_numpy()
        score = metric(y_time=y_time, y_event=y_event, y_pred=y_pred, race_group=race_group)

        return score

    @property
    def __name__(self) -> str:
        return self._name
