import config
import polars as pl

from src.feature.tabular import OrdinalEncoder, RawEncoder
from src.model.sklearn_like import TabMRegressor, TabMRegressorWrapper
from src.trainer.tabular.simple import single_inference_fn

raw_train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv").with_columns(
    pl.lit("TRAIN").alias(config.DATASET_COL),
    pl.lit(-1).alias(config.FOLD_COL),
)
raw_test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv").with_columns(pl.lit("TEST").alias(config.DATASET_COL))
train_test_df = pl.concat([raw_train_df, raw_test_df], how="diagonal_relaxed").sort(
    config.DATASET_COL, config.ID_COL, descending=[True, False]
)


def fe(
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


def preprocess(features_df: pl.DataFrame, output_dataset: None | str = "TRAIN") -> pl.DataFrame:
    from sklearn.impute import SimpleImputer

    numerical_cols = [f"{config.FEATURE_PREFIX}n_{col}" for col in config.NUMERICAL_COLS]

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(features_df.filter(pl.col(config.DATASET_COL) == "TRAIN").select(numerical_cols))

    features_df = pl.concat(
        [
            features_df.drop(numerical_cols),
            pl.DataFrame(imputer.transform(features_df.select(numerical_cols)), schema=numerical_cols),
        ],
        how="horizontal",
    )
    if not output_dataset:
        return features_df
    return features_df.filter(pl.col(config.DATASET_COL) == output_dataset)


test_features_df = fe(train_test_df, output_dataset=None)
preprocessed_test_features_df = preprocess(test_features_df, output_dataset="TEST")
feature_names = sorted([x for x in preprocessed_test_features_df.columns if x.startswith(config.FEATURE_PREFIX)])
cat_features = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}c_")]

print(f"# of features: {len(feature_names)}")
print(f"# of cat_features: {len(cat_features)}")


te_result_df = single_inference_fn(
    model=TabMRegressorWrapper(
        name="tabm2",
        model=TabMRegressor(
            categorical_features=cat_features,
            random_state=config.SEED,
        ),
        feature_names=feature_names,
    ),
    features_df=preprocessed_test_features_df,
    feature_names=feature_names,
    model_dir=config.ARTIFACT_EXP_DIR(),
    inference_folds=list(range(config.N_SPLITS)),
    out_dir=config.OUTPUT_DIR / config.EXP_NAME if config.IS_KAGGLE_ENV else config.OUTPUT_DIR,
)
print(config.ARTIFACT_EXP_DIR(), config.ARTIFACT_EXP_DIR().exists())
print(te_result_df["pred"].to_list())

te_result_df.select([config.ID_COL, "pred"]).rename({"pred": "prediction"}).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
