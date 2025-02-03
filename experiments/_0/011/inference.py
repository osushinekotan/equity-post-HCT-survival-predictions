import config
import polars as pl

from src.model.sklearn_like import WeightedAverageModel, WeightedAverageModelWrapper
from src.trainer.tabular.simple import single_inference_fn


def load_ensemble_source(
    train_test_df: pl.DataFrame,
    exp_names: dict[str, str],
    source_prefix: str = "",
    output_dataset: str = "TRAIN",
) -> pl.DataFrame:
    te_result_path = (  # noqa
        lambda exp_name, model_name: config.ARTIFACT_EXP_DIR(exp_name) / model_name / "te_result.csv"
        if not config.IS_KAGGLE_ENV
        else config.OUTPUT_DIR / exp_name / model_name / "te_result.csv"
    )

    source_dfs = [
        pl.concat(
            [
                pl.read_csv(
                    config.ARTIFACT_EXP_DIR(exp_name) / model_name / "va_result.csv",
                    columns=[config.ID_COL, "pred"],
                )
                .with_columns(pl.col("pred").rank().alias(f"pred_rank_{exp_name}"))
                .rename({"pred": f"pred_{exp_name}"}),
                pl.read_csv(te_result_path(exp_name, model_name), columns=[config.ID_COL, "pred"])
                .with_columns(pl.col("pred").rank().alias(f"pred_rank_{exp_name}"))
                .rename({"pred": f"pred_{exp_name}"}),
            ],
            how="diagonal_relaxed",
        )
        for exp_name, model_name in exp_names.items()
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


raw_train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv").with_columns(
    pl.lit("TRAIN").alias(config.DATASET_COL),
    pl.lit(-1).alias(config.FOLD_COL),
)
raw_test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv").with_columns(pl.lit("TEST").alias(config.DATASET_COL))
train_test_df = pl.concat([raw_train_df, raw_test_df], how="diagonal_relaxed").sort(
    config.DATASET_COL, config.ID_COL, descending=[True, False]
)

features_df = load_ensemble_source(
    train_test_df=train_test_df,
    exp_names=config.ENSEMBLE_EXP_NAMES,
    source_prefix=config.FEATURE_PREFIX,
    output_dataset="TEST",
)

feature_names = sorted([x for x in features_df.columns if x.startswith(config.FEATURE_PREFIX)])
feature_names = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}pred_rank_")]

cat_features = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}c_")]

print(f"# of features: {len(feature_names)}")
print(f"# of cat_features: {len(cat_features)}")


te_result_df = single_inference_fn(
    model=WeightedAverageModelWrapper(
        name="ave",
        model=WeightedAverageModel(weights=None),
        feature_names=feature_names,
    ),
    features_df=features_df,
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
