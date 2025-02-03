import config
import polars as pl

from src.feature.tabular import OrdinalEncoder, RawEncoder
from src.model.sklearn_like import CatBoostRegressorWrapper
from src.trainer.tabular.simple import single_inference_fn

raw_train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv").with_columns(
    pl.lit("TRAIN").alias(config.DATASET_COL),
    pl.lit(-1).alias(config.FOLD_COL),
)
raw_test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv").with_columns(pl.lit("TEST").alias(config.DATASET_COL))
train_test_df = pl.concat([raw_train_df, raw_test_df], how="diagonal_relaxed").sort(
    config.DATASET_COL, config.ID_COL, descending=[True, False]
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
    OrdinalEncoder(
        columns=(
            [
                *config.CATEGORICAL_COLS,
            ]
        ),
        prefix=f"{config.FEATURE_PREFIX}c_",
    ),
]

train_df = train_test_df.filter(pl.col(config.DATASET_COL) == "TRAIN")
for encoder in encoders:
    encoder.fit(train_df)

features_df = pl.concat(
    [encoder.transform(train_test_df.filter(pl.col(config.DATASET_COL) == "TEST")) for encoder in encoders],
    how="horizontal",
)

feature_names = sorted([x for x in features_df.columns if x.startswith(config.FEATURE_PREFIX)])
cat_features = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}c_")]

print(f"# of features: {len(feature_names)}")
print(f"# of cat_features: {len(cat_features)}")


# te_result_df = single_inference_fn(
#     model=CatBoostRegressorWrapper(
#         name="cat2",
#         feature_names=feature_names,
#         cat_features=cat_features,
#     ),
#     features_df=features_df,
#     feature_names=feature_names,
#     model_dir=config.ARTIFACT_EXP_DIR(),
#     inference_folds=list(range(config.N_SPLITS)),
#     out_dir=config.OUTPUT_DIR / config.EXP_NAME if config.IS_KAGGLE_ENV else config.OUTPUT_DIR,
# )
# print(config.ARTIFACT_EXP_DIR(), config.ARTIFACT_EXP_DIR().exists())
# print(te_result_df["pred"].to_list())

all_te_result_df = pl.DataFrame()
base_name = "cat2"
for i_fold in range(config.N_SPLITS):
    name = f"cat2_{i_fold}"
    te_result_df = single_inference_fn(
        model=CatBoostRegressorWrapper(
            name=f"{base_name}_{i_fold}",
            feature_names=feature_names,
            cat_features=cat_features,
        ),
        features_df=features_df,
        feature_names=feature_names,
        model_dir=config.ARTIFACT_EXP_DIR(),
        inference_folds=list(range(config.N_SPLITS)),
        out_dir=config.OUTPUT_DIR / config.EXP_NAME if config.IS_KAGGLE_ENV else config.OUTPUT_DIR,
    )
    all_te_result_df = pl.concat(
        [
            all_te_result_df,
            te_result_df.select(
                pl.col(config.ID_COL),
                pl.col("pred"),
            ),
        ],
        how="diagonal_relaxed",
    )
all_te_result_df = all_te_result_df.group_by(config.ID_COL).agg(pl.col("pred").mean().alias("pred")).sort(config.ID_COL)

out_dir = config.OUTPUT_DIR / config.EXP_NAME if config.IS_KAGGLE_ENV else config.OUTPUT_DIR
te_result_filepath = out_dir / base_name / "te_result.csv"
te_result_filepath.parent.mkdir(parents=True, exist_ok=True)
all_te_result_df.write_csv(te_result_filepath)
print(all_te_result_df["pred"].to_list())


all_te_result_df.select([config.ID_COL, "pred"]).rename({"pred": "prediction"}).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
