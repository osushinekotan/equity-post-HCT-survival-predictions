import fire
import kaggle
import rootutils

rootutils.setup_root(search_from=".", dotenv=True, pythonpath=True)

from src.kaggle_utils.dataset import download_kaggle_competition_dataset

kaggle_client = kaggle.KaggleApi()
kaggle_client.authenticate()


def main(force: bool = False, out_dir: str = "data/inputs") -> None:
    download_kaggle_competition_dataset(
        client=kaggle_client,
        competition="equity-post-HCT-survival-predictions",
        out_dir=out_dir,
        force=force,
    )


if __name__ == "__main__":
    fire.Fire(main)
