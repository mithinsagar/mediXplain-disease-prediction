"""
Download the base Kaggle dataset used by MediXplain.

Requires the `kaggle` CLI to be installed and configured with a valid
`~/.kaggle/kaggle.json` credentials file. See
https://www.kaggle.com/docs/api for setup instructions.

Author:  Mithin Sagar S
GitHub:  https://github.com/mithinsagar
Project: MediXplain - Explainable Disease Prediction System
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from config import CONFIG
from utils import get_logger

log = get_logger(__name__)

DATASET_SLUG = "dhivyeshrk/diseases-and-symptoms-dataset"
EXPECTED_FILE = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def main() -> None:
    raw_dir: Path = CONFIG.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_csv = raw_dir / EXPECTED_FILE
    if target_csv.exists():
        log.info("Dataset already present: %s", target_csv)
        return

    if not _kaggle_available():
        log.error(
            "The `kaggle` CLI is not installed. Install it with `pip install "
            "kaggle` and configure ~/.kaggle/kaggle.json before running this "
            "script, or download the CSV manually from "
            "https://www.kaggle.com/datasets/%s and place it at %s.",
            DATASET_SLUG,
            target_csv,
        )
        sys.exit(1)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(raw_dir),
        "--unzip",
    ]
    log.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)

    if not target_csv.exists():
        log.warning(
            "Downloaded successfully but expected CSV %s was not found. "
            "Inspect %s to see what was extracted.",
            EXPECTED_FILE,
            raw_dir,
        )
        return
    log.info("Dataset ready at %s", target_csv)


if __name__ == "__main__":
    main()
