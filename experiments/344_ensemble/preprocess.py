import polars as pl
from sklearn.model_selection import train_test_split

from src.feature.tabular import OrdinalEncoder, RawEncoder


def load_data(config, valid_ratio: float | None = None) -> pl.DataFrame:
    raw_train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv").with_columns(
        pl.lit("TRAIN").alias(config.DATASET_COL),
        pl.lit(-1).alias(config.FOLD_COL),
    )
    if valid_ratio is not None:
        _tr_df, _va_df = train_test_split(raw_train_df, test_size=valid_ratio, random_state=config.SEED)
        _va_df = _va_df.with_columns(
            pl.lit("VALID").alias(config.DATASET_COL),
        )
        raw_train_df = pl.concat([_tr_df, _va_df], how="diagonal_relaxed")

    raw_test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv").with_columns(
        pl.lit("TEST").alias(config.DATASET_COL)
    )

    train_test_df = pl.concat([raw_train_df, raw_test_df], how="diagonal_relaxed").sort(
        config.DATASET_COL, config.ID_COL, descending=[True, False]
    )
    return train_test_df


def fe(
    config,
    train_test_df: pl.DataFrame,
    output_dataset: str | None = "TRAIN",
) -> pl.DataFrame:
    encoders = [
        RawEncoder(columns=config.META_COLS, prefix=""),
        RawEncoder(
            columns=(
                [
                    *config.NUMERICAL_COLS,
                ]
            ),
            prefix=f"{config.FEATURE_PREFIX}n_",
        ),
        OrdinalEncoder(
            columns=(
                [
                    *config.CATEGORICAL_COLS,
                ]
            ),
            prefix=f"{config.FEATURE_PREFIX}c_",
        ),
    ]

    for encoder in encoders:
        encoder.fit(train_test_df.filter(pl.col(config.DATASET_COL) == "TRAIN"))

    features_df = pl.concat(
        [encoder.transform(train_test_df) for encoder in encoders],
        how="horizontal",
    )
    if output_dataset is None:
        return features_df
    return features_df.filter(pl.col(config.DATASET_COL) == output_dataset)


def load_ensemble_source(
    config,
    train_test_df: pl.DataFrame,
    exp_names: dict[str, str],
    source_prefix: str = "",
    output_dataset: str = "TRAIN",
) -> pl.DataFrame:
    te_result_path = (  # noqa
        lambda exp_name: config.ARTIFACT_EXP_DIR(exp_name) / "te_result.csv"
        if not config.IS_KAGGLE_ENV
        else config.OUTPUT_DIR / exp_name / "te_result.csv"
    )

    source_dfs = [
        pl.concat(
            [
                pl.read_csv(
                    config.ARTIFACT_EXP_DIR(exp_name) / "va_result.csv",
                    columns=[config.ID_COL, "pred"],
                )
                .with_columns(
                    ((pl.col("pred").rank() - 1) / (pl.col("pred").rank().max() - 1)).alias(f"rank_pred_{exp_name}")
                )
                .rename({"pred": f"pred_{exp_name}"}),
                pl.read_csv(te_result_path(exp_name), columns=[config.ID_COL, "pred"])
                .with_columns(
                    ((pl.col("pred").rank() - 1) / (pl.col("pred").rank().max() - 1)).alias(f"rank_pred_{exp_name}")
                )
                .rename({"pred": f"pred_{exp_name}"}),
            ],
            how="diagonal_relaxed",
        )
        for exp_name in exp_names
    ]
    source_df = source_dfs.pop(0)
    for df in source_dfs:
        source_df = source_df.join(df, on=config.ID_COL, how="left")

    # add prefix
    source_df = source_df.select(
        [
            config.ID_COL,
            pl.all().exclude(config.ID_COL).name.prefix(source_prefix),
        ]
    )
    return train_test_df.join(source_df, on=config.ID_COL, how="left").filter(
        pl.col(config.DATASET_COL) == output_dataset
    )
