from kfp import dsl

@dsl.component(
        base_image="python:3.12-slim",
        packages_to_install=["kagglehub"]
)
def collect_data(output_dir: str = "data/raw"):
    """
    Download the Telco Customer Churn dataset from Kaggle
    """
    import logging
    import shutil
    import kagglehub
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    DATASET_HANDLE = "blastchar/telco-customer-churn"
    DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s", DATASET_HANDLE)

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(output_dir))

    downloaded_path = Path(downloaded_path)

    logger.info("Kaggle download location: %s", downloaded_path)

    csv_candidates = list(downloaded_path.rglob(DATASET_FILENAME))

    direct_csv = output_dir / DATASET_FILENAME

    if direct_csv.exists():
        source_csv = direct_csv
    elif csv_candidates:
        source_csv = csv_candidates[0]
    else:
        raise FileNotFoundError(
            f"Could not find {DATASET_FILENAME} after downloading "
            f"{DATASET_HANDLE}.\n"
            f"Downloaded path: {downloaded_path}")

    destination = output_dir / DATASET_FILENAME

    if source_csv.resolve() != destination.resolve():
        shutil.copy2(source_csv, destination)

    logger.info("Raw dataset saved to: %s", destination)

    return destination



if __name__ == "__main__":
    collect_data()