import logging
from pathlib import Path

import polars as pl

from src.model.sklearn_like import BaseWrapper

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

    for i_fold in train_folds:
        logger.info(f"🚀 >>> Start training fold {i_fold} =============")
        tr_x = features_df.filter(pl.col(fold_col) != i_fold).select(feature_cols).to_numpy()
        tr_y = features_df.filter(pl.col(fold_col) != i_fold)[target_col].to_numpy()

        va_df = features_df.filter(pl.col(fold_col) == i_fold)
        va_x = va_df.select(feature_cols).to_numpy()
        va_y = va_df[target_col].to_numpy()

        i_out_dir = out_dir / f"fold_{i_fold:02}"

        if use_eval_metric_extra_va_df:
            model.params["eval_metric"].va_df = va_df

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

    return va_result_df, va_scores, trained_models
