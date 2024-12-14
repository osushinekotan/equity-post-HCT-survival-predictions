import numpy as np
import polars as pl
from lifelines.utils import concordance_index


class CustomMetric:
    # https://www.kaggle.com/code/metric/eefs-concordance-index
    def __init__(
        self,
        event_label: str = "efs",
        interval_label: str = "efs_time",
        prediction_label: str = "pred",
        name: str | None = None,
    ):
        self.event_label = event_label
        self.interval_label = interval_label
        self.prediction_label = prediction_label
        self._name = name or self.__class__.__name__

    def __call__(self, input_df: pl.DataFrame) -> float:
        merged_df = input_df.to_pandas()
        merged_df_race_dict = dict(merged_df.groupby(["race_group"]).groups)
        metric_list = []
        for race in merged_df_race_dict.keys():
            # Retrieving values from y_test based on index
            indices = sorted(merged_df_race_dict[race])
            merged_df_race = merged_df.iloc[indices]
            # Calculate the concordance index
            c_index_race = concordance_index(
                merged_df_race[self.interval_label],
                -merged_df_race[self.prediction_label],
                merged_df_race[self.event_label],
            )
            metric_list.append(c_index_race)
        return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))

    @property
    def __name__(self) -> str:
        return self._name
