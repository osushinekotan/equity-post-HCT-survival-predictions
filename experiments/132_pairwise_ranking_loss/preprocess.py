import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def preprocess(config, features_df: pl.DataFrame, output_dataset: None | str = "TRAIN") -> pl.DataFrame:
    numerical_cols = [f"{config.FEATURE_PREFIX}n_{col}" for col in config.NUMERICAL_COLS]
    categorical_cols = [f"{config.FEATURE_PREFIX}c_{col}" for col in config.CATEGORICAL_COLS]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    train_df = features_df.filter(pl.col(config.DATASET_COL) == "TRAIN")
    imputer.fit(train_df.select(numerical_cols))
    scaler.fit(imputer.transform(train_df.select(numerical_cols)))

    features_df = pl.concat(
        [
            features_df.drop(numerical_cols),
            pl.DataFrame(
                scaler.transform(imputer.transform(features_df.select(numerical_cols))), schema=numerical_cols
            ),
        ],
        how="horizontal",
    )

    # categorical < 1 -> 0
    features_df = features_df.with_columns(
        [pl.when(pl.col(c) < 0).then(0).otherwise(pl.col(c)).alias(c) for c in categorical_cols]
    )

    if not output_dataset:
        return features_df
    return features_df.filter(pl.col(config.DATASET_COL) == output_dataset)
