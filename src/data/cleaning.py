from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas"],
)
def clean_data(input_path: str, validation_report: str, output_dir: str = "data/cleaned") -> str:
    """
    Clean the Telco Customer Churn dataset based on validation results

    Returns the path to the cleaned CSV
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

    NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

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

    def _parse_report(report_json: str) -> dict:
        return json.loads(report_json)

    report = _parse_report(validation_report)

    if not report["is_valid"]:
        logger.warning(
            "Validation found issues – proceeding with cleaning:\n  %s",
            "\n  ".join(report["errors"]),
        )

    logger.info("Starting cleaning")

    output_path = Path(output_dir) / "cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    logger.info("Raw shape: %s", df.shape)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID column")

    dup_count = int(df.duplicated().sum())
    if dup_count:
        df = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows", dup_count)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[c for c in NUMERIC_COLUMNS if c in df.columns])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with NaN in numeric columns", dropped)

    for col in YES_NO_COLUMNS:
        if col not in df.columns:
            continue
        mapping = {"Yes": 1, "No": 0}
        # handle "No internet service" / "No phone service" as 0
        df[col] = df[col].map(lambda v: mapping.get(v, 0)).astype(int)
        logger.info("Encoded %s (Yes/No -> 1/0)", col)

    df = pd.get_dummies(df, columns=[c for c in ONE_HOT_COLUMNS if c in df.columns])
    logger.info("One-hot encoded columns: %s", ONE_HOT_COLUMNS)

    df.to_csv(output_path, index=False)
    logger.info("Cleaned shape: %s  ->  %s", df.shape, output_path)

    return str(output_path)


if __name__ == "__main__":
    from validation import validate_data
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    report = validate_data(input_path=raw_path)
    clean_data(input_path=raw_path, validation_report=report)
