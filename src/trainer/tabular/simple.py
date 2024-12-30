import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.model.sklearn_like import BaseWrapper
from src.model.visualization import plot_feature_importance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def single_train_fn(
    model: BaseWrapper,
    features_df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    meta_cols: list[str],
    fold_col: str | Path = "fold",
    out_dir: str | Path = "./outputs",
    eval_fn: None = None,
    train_folds: list[int] | None = None,
    overwrite: bool = False,
    **kwargs,
) -> tuple[pl.DataFrame, dict[str, float], list[BaseWrapper]]:
    va_records, va_scores, trained_models = [], {}, []
    out_dir = Path(out_dir) / model.name

    train_folds = train_folds or features_df[fold_col].unique().to_list()
    use_eval_metric_extra_va_df = kwargs.get("use_eval_metric_extra_va_df", False)
    enable_plot_feature_importance = kwargs.get("enable_plot_feature_importance", True)

    for i_fold in train_folds:
        logger.info(f"🚀 >>> Start training fold {i_fold} =============")
        tr_x = features_df.filter(pl.col(fold_col) != i_fold).select(feature_cols).to_numpy()
        tr_y = features_df.filter(pl.col(fold_col) != i_fold)[target_col].to_numpy()

        va_df = features_df.filter(pl.col(fold_col) == i_fold)
        va_x = va_df.select(feature_cols).to_numpy()
        va_y = va_df[target_col].to_numpy()

        i_out_dir = out_dir / f"fold_{i_fold:02}"

        if use_eval_metric_extra_va_df:
            model.eval_metric.va_df = va_df

        if model.get_save_path(out_dir=i_out_dir).exists() and not overwrite:
            model.load(out_dir=i_out_dir)
            logger.info(f"   - ❌ Skip training fold {i_fold}")
        else:
            model.fit(tr_x=tr_x, tr_y=tr_y, va_x=va_x, va_y=va_y)
            model.save(out_dir=i_out_dir)

        trained_models.append(model)
        logger.info("   - ✅ Successfully saved model")

        va_pred = model.predict(va_x)

        try:
            if va_pred.shape[1] == 1:
                va_pred = va_pred.reshape(-1)
        except IndexError:
            pass

        i_va_df = va_df.select(meta_cols).with_columns(
            [
                pl.Series(va_pred).alias("pred"),
                pl.lit(model.name).alias("name"),
            ]
        )
        i_va_df.write_csv(i_out_dir / "va_pred.csv")
        va_records.append(i_va_df)
        logger.info("   - ✅ Successfully predicted validation data")

        i_score = eval_fn(input_df=i_va_df)
        va_scores[f"{eval_fn.__name__}_fold_{i_fold:02}"] = i_score

        logger.info(f"   - {eval_fn.__name__}: {i_score}")
        logger.info(f"🎉 <<< Finish training fold {i_fold} =============\n\n")

    va_result_df = pl.concat(va_records, how="diagonal")
    score = eval_fn(input_df=va_result_df)
    va_scores[f"{eval_fn.__name__}_full"] = score
    logger.info(f"✅ Final {eval_fn.__name__}: {score}")

    # plot feature importance
    if enable_plot_feature_importance:
        import pandas as pd

        importance_df = pd.DataFrame()
        for i, m in enumerate(trained_models):
            i_df = pd.DataFrame(
                {"feature_importance": m.feature_importances_, "feature_name": m.feature_names, "fold": i}
            )
            importance_df = pd.concat([importance_df, i_df], axis=0, ignore_index=True)

        fig = plot_feature_importance(
            df=importance_df,
            feature_name_col="feature_name",
            feature_importance_col="feature_importance",
            fold_col="fold",
            top_k=50,
        )
        fig.savefig(out_dir / "feature_importance.png", dpi=300)

    with open(out_dir / "va_scores.json", "w") as f:
        json.dump(va_scores, f, indent=4)

    va_result_df.write_csv(out_dir / "va_result.csv")

    return va_result_df, va_scores, trained_models


def single_inference_fn(
    model: BaseWrapper,
    features_df: pl.DataFrame,
    feature_names: list[str],
    model_dir: str | Path,
    inference_folds: list[int],
) -> pl.DataFrame:
    te_preds = []
    model_dir = Path(model_dir) / model.name
    for i_fold in inference_folds:
        logger.info(f"🚀 >>> Start training fold {i_fold} =============")

        te_x = features_df.select(feature_names).to_numpy()
        i_out_dir = Path(model_dir) / f"fold_{i_fold:02}"

        try:
            model.load(out_dir=i_out_dir)
            logger.info("   - ✅ Successfully loaded model")

            te_pred = model.predict(te_x)
            try:
                if te_pred.shape[1] == 1:
                    te_pred = te_pred.reshape(-1)
            except IndexError:
                pass
            te_preds.append(te_pred)
        except Exception as e:
            logger.error(f"   - ❌ Failed to load model: {e}")

    te_pred_ave = np.mean(te_preds, axis=0)
    return te_pred_ave
