import config
import polars as pl

df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")
df.write_csv(config.OUTPUT_DIR / "submission.csv")

print(config.OUTPUT_DIR)
print(pl.read_csv(config.OUTPUT_DIR / "submission.csv").shape)

print(config.ARTIFACT_DIR)
print(pl.read_csv(config.ARTIFACT_DIR / "submission.csv").shape)

if not config.IS_KAGGLE_ENV:
    from src.kaggle_utils.dataset import dataset_upload

    dataset_upload(
        handle=config.ARTIFACTS_HANDLE,
        local_dataset_dir=config.OUTPUT_DIR,
        update=True,
    )
    dataset_upload(
        handle=config.CODES_HANDLE,
        local_dataset_dir=config.ROOT_DIR,
        update=True,
    )
