import config
import polars as pl
from preprocess import fe, load_data

from src.model.sklearn_like import CatBoostRegressorWrapper
from src.trainer.tabular.simple import single_inference_fn

train_test_df = load_data(config=config, valid_ratio=None)

features_df = fe(config=config, train_test_df=train_test_df, output_dataset="TEST")
feature_names = sorted([x for x in features_df.columns if x.startswith(config.FEATURE_PREFIX)])
cat_features = [x for x in feature_names if x.startswith(f"{config.FEATURE_PREFIX}c_")]

te_result_df = pl.DataFrame()
out_dir = config.OUTPUT_DIR / config.EXP_NAME if config.IS_KAGGLE_ENV else config.OUTPUT_DIR

for seed in config.SEEDS:
    name = f"cat_{seed}"

    _te_result_df = single_inference_fn(
        model=CatBoostRegressorWrapper(name=name, feature_names=feature_names, cat_features=cat_features),
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
