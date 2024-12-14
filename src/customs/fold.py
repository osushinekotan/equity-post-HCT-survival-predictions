import numpy as np
import polars as pl
from sklearn.model_selection import KFold


def add_kfold(
    input_df: pl.DataFrame,
    n_splits: int,
    random_state: int,
    fold_col: str,
) -> pl.DataFrame:
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.zeros(len(input_df), dtype=np.int32)
    for fold, (_, valid_idx) in enumerate(skf.split(X=input_df)):
        folds[valid_idx] = fold
    return input_df.with_columns(pl.Series(name=fold_col, values=folds))
