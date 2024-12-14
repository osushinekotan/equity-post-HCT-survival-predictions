import json
import logging
import os
import re
import shutil
import tempfile
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path

import fire
import rootutils
from kaggle import KaggleApi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT = rootutils.setup_root(".")

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")


class Deploy:
    """Push experiment outputs to kaggle dataset."""

    default_ignore_patterns = [".git", "__pycache__", "*.pyc", "*.ipynb_checkpoints"]

    def __init__(
        self,
        client: "KaggleApi",
        artifact_dataset_name: str | None,
        code_dataset_name: str | None,
    ):
        self.client = client
        self.artifact_dataset_name = artifact_dataset_name
        self.code_dataset_name = code_dataset_name

    def push_artifact(self, source_dir: str) -> None:
        """Push output directory to kaggle dataset."""
        if self.artifact_dataset_name is None:
            raise ValueError("artifact_dataset_name is required.")

        # model and predictions
        metadata = make_dataset_metadata(dataset_name=self.artifact_dataset_name)

        # if exist dataset, stop pushing
        if exist_dataset(
            dataset=f"{KAGGLE_USERNAME}/{self.artifact_dataset_name}",
            existing_dataset=self.existing_dataset,
        ):
            logger.warning(f"{self.artifact_dataset_name} already exist!! Stop pushing. 🛑")
            return

        with tempfile.TemporaryDirectory() as tempdir:
            dst_dir = Path(tempdir) / self.artifact_dataset_name

            copytree(
                src=str(source_dir),
                dst=str(dst_dir),
                ignore_patterns=self.default_ignore_patterns,
            )
            self._display_tree(dst_dir=dst_dir)

            with open(Path(dst_dir) / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            self.client.dataset_create_new(
                folder=dst_dir,
                public=False,
                quiet=False,
                dir_mode="zip",
            )

    def push_huguingface_model(self, model_name: str | None = None) -> None:
        """Push huggingface model and transformer to kaggle dataset."""
        from transformers import AutoConfig, AutoTokenizer

        if model_name is None:
            raise ValueError("model_name is required.")

        dataset_name = re.sub(r"[/_]", "-", model_name)

        # if exist dataset, stop pushing
        if exist_dataset(
            dataset=f'{os.getenv("KAGGLE_USERNAME")}/{dataset_name}',
            existing_dataset=self.existing_dataset,
        ):
            logger.warning(f"{dataset_name} already exist!! Stop pushing. 🛑")
            return

        # pretrained tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        metadata = make_dataset_metadata(dataset_name=dataset_name)

        with tempfile.TemporaryDirectory() as tempdir:
            config.save_pretrained(tempdir)
            tokenizer.save_pretrained(tempdir)
            with open(Path(tempdir) / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            self.client.dataset_create_new(
                folder=tempdir,
                public=True,
                quiet=False,
                dir_mode="zip",
            )

    def push_code(self, src_dir: str) -> None:
        """Push source code to kaggle dataset."""
        if self.code_dataset_name is None:
            raise ValueError("code_dataset_name is required.")

        metadata = make_dataset_metadata(dataset_name=self.code_dataset_name)

        with tempfile.TemporaryDirectory() as tempdir:
            dst_dir = Path(tempdir) / self.code_dataset_name

            # for src directory
            dst_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(ROOT / "README.md", dst_dir)

            copytree(
                src=(ROOT / src_dir).as_posix(),
                dst=str(dst_dir / src_dir),
                ignore_patterns=self.default_ignore_patterns,
            )
            self._display_tree(dst_dir=dst_dir)

            with open(dst_dir / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

            # update dataset if dataset already exist
            if exist_dataset(
                dataset=f"{KAGGLE_USERNAME}/{self.code_dataset_name}",
                existing_dataset=self.existing_dataset,
            ):
                logger.info("update code")
                self.client.dataset_create_version(
                    folder=dst_dir,
                    version_notes="latest",
                    quiet=False,
                    convert_to_csv=False,
                    delete_old_versions=True,
                    dir_mode="zip",
                )
            else:
                logger.info("create dataset of code")
                self.client.dataset_create_new(
                    folder=dst_dir,
                    public=False,
                    quiet=False,
                    dir_mode="zip",
                )

    @cached_property
    def existing_dataset(self) -> list:
        """Check existing dataset in kaggle."""
        return self.client.dataset_list(user=os.getenv("KAGGLE_USERNAME"))

    @staticmethod
    def _display_tree(dst_dir: Path) -> None:
        logger.info(f"dst_dir={dst_dir}\ntree")
        display_tree(dst_dir)


def exist_dataset(dataset: str, existing_dataset: list) -> bool:
    """Check if dataset already exist in kaggle."""
    for ds in existing_dataset:
        if str(ds) == dataset:
            return True
    return False


def make_dataset_metadata(dataset_name: str) -> dict:
    """Make metadata of kaggle dataset."""
    dataset_metadata = {}
    dataset_metadata["id"] = f'{os.environ["KAGGLE_USERNAME"]}/{dataset_name}'
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]  # type: ignore
    dataset_metadata["title"] = dataset_name
    return dataset_metadata


def copytree(src: str, dst: str, ignore_patterns: list | None = None) -> None:
    """Copytree with ignore patterns."""
    ignore_patterns = ignore_patterns or []

    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        if any(fnmatch(item, pattern) for pattern in ignore_patterns):
            continue

        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, ignore_patterns)
        else:
            shutil.copy2(s, d)


def display_tree(directory: Path, file_prefix: str = "") -> None:
    """Display directory tree."""
    entries = list(directory.iterdir())
    file_count = len(entries)

    for i, entry in enumerate(sorted(entries, key=lambda x: x.name)):
        if i == file_count - 1:
            prefix = "└── "
            next_prefix = file_prefix + "    "
        else:
            prefix = "├── "
            next_prefix = file_prefix + "│   "

        line = file_prefix + prefix + entry.name
        print(line)

        if entry.is_dir():
            display_tree(entry, next_prefix)


def main(
    artifact_dataset_name: str | None = None,
    code_dataset_name: str | None = None,
    source_dir: str | None = None,
    src_dir: str | None = None,
    hf_model_name: str | None = None,
) -> None:
    """Push experiment outputs to kaggle dataset."""
    client = KaggleApi()
    client.authenticate()

    deploy = Deploy(
        client=client,
        artifact_dataset_name=artifact_dataset_name,
        code_dataset_name=code_dataset_name,
    )

    if source_dir:
        deploy.push_artifact(source_dir=source_dir)
    if src_dir:
        deploy.push_code(src_dir=src_dir)
    if hf_model_name:
        deploy.push_huguingface_model(model_name=hf_model_name)


if __name__ == "__main__":
    print(ROOT)
    fire.Fire(main)
