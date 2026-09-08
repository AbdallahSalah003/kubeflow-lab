from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas"],
)
def clean_data(
    input_dataset: Input[Dataset],
    validation_report: Input[Artifact],
    output_dataset: Output[Dataset],
):
    """
    Clean the Telco Customer Churn dataset.
    """
    import json
    import logging
    from pathlib import Path

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    NUMERIC_COLUMNS = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    YES_NO_COLUMNS = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "MultipleLines",
    ]

    ONE_HOT_COLUMNS = [
        "gender",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]

    # Read validation report.
    report_path = Path(validation_report.path)

    report = json.loads(report_path.read_text())

    if not report["is_valid"]:
        raise ValueError(
            "Cannot clean invalid dataset. " f"Validation errors: {report['errors']}"
        )

    logger.info("Validation passed. Starting cleaning.")

    input_path = Path(input_dataset.path)

    df = pd.read_csv(input_path)

    logger.info(
        "Raw shape: %s",
        df.shape,
    )

    # Remove ID.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Remove duplicate rows.
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        df = df.drop_duplicates()

        logger.info(
            "Dropped %d duplicate rows",
            duplicate_count,
        )

    # Convert numeric columns.
    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Handle missing numeric values.
    before = len(df)

    df = df.dropna(
        subset=[column for column in NUMERIC_COLUMNS if column in df.columns]
    )

    dropped = before - len(df)

    if dropped:
        logger.info(
            "Dropped %d rows with missing numeric values",
            dropped,
        )

    # Encode Yes/No columns.
    for column in YES_NO_COLUMNS:

        if column not in df.columns:
            continue

        mapping = {
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
            "No phone service": 0,
        }

        df[column] = df[column].map(mapping).astype(int)

    # One-hot encode remaining categorical columns.
    one_hot_columns = [column for column in ONE_HOT_COLUMNS if column in df.columns]

    df = pd.get_dummies(
        df,
        columns=one_hot_columns,
        dtype=int,
    )

    logger.info(
        "One-hot encoded: %s",
        one_hot_columns,
    )

    if df.isna().any().any():
        raise ValueError("Cleaned dataset still contains missing values.")

    output_path = Path(output_dataset.path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Cleaned dataset saved to: %s",
        output_path,
    )

    logger.info(
        "Cleaned shape: %s",
        df.shape,
    )
