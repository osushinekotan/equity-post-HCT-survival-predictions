import json

import config
import polars as pl
from preprocess import fe, load_data

from src.model.sklearn_like import CatBoostClassifierWrapper, CatBoostRegressorWrapper, LightGBMWapper
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

all_te_result_df = pl.DataFrame()
for seed in [0, 1, 2, 3, 4]:
    _te_result_df = single_inference_fn(
        model=LightGBMWapper(name=f"lgb1_{seed}"),
        features_df=features_df,
        feature_names=feature_names,
        model_dir=config.ARTIFACT_EXP_DIR(),
        inference_folds=list(range(config.N_SPLITS)),
        out_dir=out_dir,
    )
    all_te_result_df = pl.concat(
        [
            all_te_result_df,
            _te_result_df.select(
                [
                    pl.col(config.ID_COL),
                    pl.col("pred"),
                ]
            ),
        ],
        how="diagonal_relaxed",
    )
    _te_result_df = single_inference_fn(
        model=CatBoostClassifierWrapper(
            name=f"cat1_{seed}",
            feature_names=feature_names,
            cat_features=cat_features,
        ),
        features_df=features_df,
        feature_names=feature_names,
        model_dir=config.ARTIFACT_EXP_DIR(),
        inference_folds=list(range(config.N_SPLITS)),
        out_dir=out_dir,
    )
    all_te_result_df = pl.concat(
        [
            all_te_result_df,
            _te_result_df.select(
                [
                    pl.col(config.ID_COL),
                    pl.col("pred"),
                ]
            ),
        ],
        how="diagonal_relaxed",
    )

agg_te_result_df = all_te_result_df.group_by(config.ID_COL).agg(pl.col("pred").mean().alias("t_event_pred")).sort("ID")
print(agg_te_result_df)

te_result_agg_df = te_result_agg_df.join(
    agg_te_result_df,
    on=config.ID_COL,
    how="left",
)
print(te_result_agg_df)

best_params = {"threshold": 0.8061224489795917, "adjustment": 0.09591836734693879, "score": 0.6946843832311068}
print(best_params)

# post process
te_result_agg_df = te_result_agg_df.with_columns(
    (
        (pl.when(pl.col("t_event_pred") < best_params["threshold"]))
        .then(pl.col("pred"))
        .otherwise(pl.col("pred") + best_params["adjustment"])
    ).alias("pred")
)
print(te_result_agg_df)

print(config.ARTIFACT_EXP_DIR(), config.ARTIFACT_EXP_DIR().exists())
print(te_result_agg_df["pred"].to_list())

te_result_agg_df.write_csv(out_dir / "te_result.csv")

# make submission
te_result_agg_df.select([config.ID_COL, "pred"]).rename({"pred": "prediction"}).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
