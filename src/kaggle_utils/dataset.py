import logging
import os
import subprocess
from pathlib import Path

from kaggle import KaggleApi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")


def download_kaggle_competition_dataset(
    client: "KaggleApi",
    competition: str,
    out_dir: Path | str,
    force: bool = False,
) -> None:
    """Download kaggle competition dataset.

    Args:
    ----
        client (KaggleApi):
        competition (str):
        out_dir (Path): destination directory
        force (bool, optional): if True, overwrite existing dataset. Defaults to False.

    """
    out_dir = Path(out_dir) / competition
    zipfile_path = out_dir / f"{competition}.zip"
    zipfile_path.parent.mkdir(exist_ok=True, parents=True)

    if not zipfile_path.is_file() or force:
        client.competition_download_files(
            competition=competition,
            path=out_dir,
            quiet=False,
            force=force,
        )

        subprocess.run(["unzip", "-o", "-q", zipfile_path, "-d", out_dir])
    else:
        logger.info(f"Dataset ({competition}) already exists.")


def download_kaggle_datasets(
    client: "KaggleApi",
    datasets: list[str],
    out_dir: Path | str,
    force: bool = False,
) -> None:
    """Download kaggle datasets.

    Args:
    ----
        client (KaggleApi):
        datasets (list[str]): list of dataset names (e.g. ["username/dataset-name"])
        out_dir (Path): destination directory
        force (bool, optional): if True, overwrite existing dataset. Defaults to False.
    """
    for dataset in datasets:
        dataset_name = dataset.split("/")[1]
        out_dir = Path(out_dir) / dataset_name
        zipfile_path = out_dir / f"{dataset_name}.zip"

        out_dir.mkdir(exist_ok=True, parents=True)

        if not zipfile_path.is_file() or force:
            logger.info(f"Downloading dataset: {dataset}")
            client.dataset_download_files(
                dataset=dataset,
                quiet=False,
                unzip=False,
                path=out_dir,
                force=force,
            )

            subprocess.run(["unzip", "-o", "-q", zipfile_path, "-d", out_dir])
        else:
            logger.info(f"Dataset ({dataset}) already exists.")
