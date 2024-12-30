import logging

import config
import polars as pl

from src.feature.tabular import RawEncoder
from src.model.sklearn_like import LightGBMWapper
from src.trainer.tabular.simple import single_inference_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

raw_train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv").with_columns(
    pl.lit("TRAIN").alias(config.DATASET_COL),
    pl.lit(-1).alias(config.FOLD_COL),
)
raw_test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv").with_columns(pl.lit("TEST").alias(config.DATASET_COL))

train_test_df = pl.concat([raw_train_df, raw_test_df], how="diagonal_relaxed")
train_test_df = train_test_df.with_columns(
    pl.col(config.CATEGORICAL_COLS)
    # .fill_null("NAN")
    .cast(pl.Categorical)
    .to_physical()
    .cast(pl.Int32)
)


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
    RawEncoder(
        columns=(
            [
                *config.CATEGORICAL_COLS,
            ]
        ),
        prefix=f"{config.FEATURE_PREFIX}c_",
    ),
]

for encoder in encoders:
    encoder.fit(raw_train_df)

features_df = pl.concat(
    [encoder.transform(train_test_df.filter(pl.col(config.DATASET_COL) == "TEST")) for encoder in encoders],
    how="horizontal",
)

feature_names = sorted([x for x in features_df.columns if x.startswith(config.FEATURE_PREFIX)])
cat_features = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}c_")]

logger.info(f"# of features: {len(feature_names)}")
logger.info(f"# of cat_features: {len(cat_features)}")


test_preds = single_inference_fn(
    model=LightGBMWapper(name="lgb"),
    features_df=features_df,
    feature_names=feature_names,
    model_dir=config.ARTIFACT_EXP_DIR(),
    inference_folds=list(range(config.N_SPLITS)),
)
features_df.select(config.ID_COL).with_columns(pl.Series("prediction", test_preds)).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
