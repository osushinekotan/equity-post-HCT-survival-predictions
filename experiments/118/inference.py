import config
import polars as pl
from preprocess import load_data, load_ensemble_source
from sklearn import linear_model

from src.model.sklearn_like import LinearWrapper
from src.trainer.tabular.simple import single_inference_fn

train_test_df = load_data(config=config, valid_ratio=None)
features_df = load_ensemble_source(
    config=config,
    train_test_df=train_test_df,
    exp_names=config.ENSEMBLE_EXP_NAMES,
    source_prefix=config.FEATURE_PREFIX,
    output_dataset="TEST",
)


feature_names = sorted([x for x in features_df.columns if x.startswith(config.FEATURE_PREFIX)])
feature_names = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}rank_pred_")]
cat_features = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}c_")]

te_result_df = pl.DataFrame()
out_dir = config.OUTPUT_DIR / config.EXP_NAME if config.IS_KAGGLE_ENV else config.OUTPUT_DIR

for seed in config.SEEDS:
    name = f"ridge_{seed}"

    _te_result_df = single_inference_fn(
        model=LinearWrapper(
            name=name,
            model=linear_model.Ridge(),
            scaling=False,
            feature_names=feature_names,
        ),
        features_df=features_df,
        feature_names=feature_names,
        model_dir=config.ARTIFACT_EXP_DIR(),
        inference_folds=list(range(config.N_SPLITS)),
        out_dir=out_dir,
    )
    te_result_df = pl.concat([te_result_df, _te_result_df], how="diagonal_relaxed")

te_result_agg_df = (
    te_result_df.group_by(config.ID_COL)
    .agg(pl.col("pred").mean())
    .sort("ID")
    .join(train_test_df.select(config.META_COLS), on=config.ID_COL, how="left")
)


print(config.ARTIFACT_EXP_DIR(), config.ARTIFACT_EXP_DIR().exists())
print(te_result_agg_df["pred"].to_list())

te_result_agg_df.write_csv(out_dir / "te_result.csv")

# make submission
te_result_agg_df.select([config.ID_COL, "pred"]).rename({"pred": "prediction"}).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
