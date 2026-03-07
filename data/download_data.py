"""
download_data.py — Automated Dataset Downloader
==================================================
WHY THIS EXISTS:
    The data/ folder is intentionally empty in the repo (datasets are
    too large for Git). This script fetches all required datasets
    automatically using the Kaggle API.

PREREQUISITES:
    1. Install kaggle: pip install kaggle
    2. Get your Kaggle API token:
       - Go to kaggle.com → Your Profile → Account → Create New Token
       - This downloads kaggle.json
       - Place it in: C:/Users/<YourName>/.kaggle/kaggle.json
    3. Run: python data/download_data.py

WHAT IT DOWNLOADS:
    1. Give Me Some Credit (150K rows) → data/raw/cs-training.csv
    2. LendingClub (2.2M rows) → data/raw/accepted_2007_to_2018Q4.csv
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_gmsc():
    """
    Download the Give Me Some Credit dataset from Kaggle.

    Source: https://www.kaggle.com/competitions/GiveMeSomeCredit
    Files: cs-training.csv (~29 MB), cs-test.csv (~14 MB)

    This is the PROTOTYPING dataset (150K rows). Used in Week 1 to
    build and validate the entire pipeline before scaling to LendingClub.
    """
    output_file = RAW_DIR / "cs-training.csv"
    if output_file.exists():
        logger.info("✓ GMSC dataset already exists at %s", output_file)
        return

    logger.info("Downloading Give Me Some Credit dataset...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.competition_download_files(
            "GiveMeSomeCredit",
            path=str(RAW_DIR),
            quiet=False,
        )

        # Extract zip if downloaded as archive
        zip_path = RAW_DIR / "GiveMeSomeCredit.zip"
        if zip_path.exists():
            logger.info("Extracting %s...", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(RAW_DIR))
            zip_path.unlink()

        logger.info("✓ GMSC dataset downloaded → %s", RAW_DIR)

    except ImportError:
        logger.error("❌ kaggle package not installed. Run: pip install kaggle")
        _print_manual_instructions("GiveMeSomeCredit")
    except Exception as e:
        logger.error("❌ Download failed: %s", e)
        _print_manual_instructions("GiveMeSomeCredit")


def download_lending_club():
    """
    Download the LendingClub dataset from Kaggle.

    Source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
    Files: accepted_2007_to_2018Q4.csv (~1.8 GB)

    This is the PRODUCTION dataset (2.2M rows). Used in Week 2 to
    retrain the full PD model at scale.
    """
    output_file = RAW_DIR / "accepted_2007_to_2018Q4.csv"
    if output_file.exists():
        logger.info("✓ LendingClub dataset already exists at %s", output_file)
        return

    logger.info("Downloading LendingClub dataset...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            "wordsforthewise/lending-club",
            path=str(RAW_DIR),
            quiet=False,
        )

        # Extract zip
        zip_path = RAW_DIR / "lending-club.zip"
        if zip_path.exists():
            logger.info("Extracting %s (this may take a minute)...", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(RAW_DIR))
            zip_path.unlink()

        logger.info("✓ LendingClub dataset downloaded → %s", RAW_DIR)

    except ImportError:
        logger.error("❌ kaggle package not installed. Run: pip install kaggle")
        _print_manual_instructions("LendingClub")
    except Exception as e:
        logger.error("❌ Download failed: %s", e)
        _print_manual_instructions("LendingClub")


def _print_manual_instructions(dataset_name: str):
    """Print manual download instructions as fallback."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  MANUAL DOWNLOAD INSTRUCTIONS")
    logger.info("=" * 60)

    if dataset_name == "GiveMeSomeCredit":
        logger.info("1. Go to: https://www.kaggle.com/competitions/GiveMeSomeCredit/data")
        logger.info("2. Download 'cs-training.csv'")
        logger.info("3. Place it in: %s", RAW_DIR)
    elif dataset_name == "LendingClub":
        logger.info("1. Go to: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        logger.info("2. Download the dataset")
        logger.info("3. Extract 'accepted_2007_to_2018Q4.csv' to: %s", RAW_DIR)

    logger.info("=" * 60)


def verify_data():
    """Check which datasets are available and print status."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  DATASET STATUS")
    logger.info("=" * 60)

    datasets = {
        "Give Me Some Credit (Prototyping)": RAW_DIR / "cs-training.csv",
        "LendingClub (Production)": RAW_DIR / "accepted_2007_to_2018Q4.csv",
    }

    all_present = True
    for name, path in datasets.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("  ✓ %-40s  %.1f MB", name, size_mb)
        else:
            logger.info("  ✗ %-40s  NOT FOUND", name)
            all_present = False

    logger.info("=" * 60)
    if all_present:
        logger.info("All datasets ready! Run the pipeline:")
        logger.info("  python src/ingestion/run_pipeline.py --dataset gmsc")
    else:
        logger.info("Some datasets missing. Run this script again or download manually.")


def main():
    """Download all datasets and verify."""
    logger.info("Risk Intelligence Engine — Data Downloader")
    logger.info("")

    download_gmsc()
    download_lending_club()
    verify_data()


if __name__ == "__main__":
    main()
