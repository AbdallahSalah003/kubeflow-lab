from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(base_image="python:3.12-slim", packages_to_install=["kagglehub"])
def collect_data(output_dataset: Output[Dataset]):
    """
    Download the Telco Customer Churn dataset from Kaggle.

    Produces:
        output_dataset: Raw Telco Customer Churn CSV.
    """
    import logging
    import shutil
    from pathlib import Path

    import kagglehub

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    DATASET_HANDLE = "blastchar/telco-customer-churn"
    DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    download_dir = Path(output_dataset.path).parent / "kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s", DATASET_HANDLE)

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(download_dir),
    )

    downloaded_path = Path(downloaded_path)

    logger.info(
        "Kaggle download location: %s",
        downloaded_path,
    )

    csv_candidates = list(downloaded_path.rglob(DATASET_FILENAME))

    if not csv_candidates:
        raise FileNotFoundError(
            f"Could not find {DATASET_FILENAME} after downloading "
            f"{DATASET_HANDLE}. Downloaded path: {downloaded_path}"
        )

    source_csv = csv_candidates[0]

    destination = Path(output_dataset.path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_csv, destination)

    logger.info(
        "Raw dataset saved to: %s",
        destination,
    )
