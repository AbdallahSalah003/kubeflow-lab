from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas"],
)
def validate_data(input_path: str) -> str:
    """
    Validate the Telco Customer Churn dataset for schema correctness,
    missing values, and data quality issues

    Returns a JSON summary of the validation results
    """
    import json
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    REQUIRED_COLUMNS = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    CATEGORICAL_VALID_VALUES = {
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "MultipleLines": {"Yes", "No", "No phone service"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "OnlineSecurity": {"Yes", "No", "No internet service"},
        "OnlineBackup": {"Yes", "No", "No internet service"},
        "DeviceProtection": {"Yes", "No", "No internet service"},
        "TechSupport": {"Yes", "No", "No internet service"},
        "StreamingTV": {"Yes", "No", "No internet service"},
        "StreamingMovies": {"Yes", "No", "No internet service"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "PaperlessBilling": {"Yes", "No"},
        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        "Churn": {"Yes", "No"},
    }

    NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    errors = []
    warnings = []

    logger.info("Loading dataset from: %s", input_path)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset from {input_path}: {e}")

    logger.info("Dataset shape: %s rows x %s columns", df.shape[0], df.shape[1])

    actual_columns = set(df.columns)
    missing = set(REQUIRED_COLUMNS) - actual_columns
    extra = actual_columns - set(REQUIRED_COLUMNS)

    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if extra:
        warnings.append(f"Unexpected columns found: {sorted(extra)}")

    if df.empty:
        errors.append("Dataset is empty (0 rows)")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate rows found")

    for col in REQUIRED_COLUMNS:
        if col in actual_columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                warnings.append(f"Column '{col}' has {null_count} null values")

    for col, valid_values in CATEGORICAL_VALID_VALUES.items():
        if col not in actual_columns:
            continue
        invalid = set(df[col].dropna().unique()) - valid_values
        if invalid:
            errors.append(
                f"Column '{col}' has invalid values: {sorted(invalid)}"
            )

    for col in NUMERIC_COLUMNS:
        if col not in actual_columns:
            continue
        non_numeric = pd.to_numeric(df[col], errors="coerce").isnull() & df[col].notna()
        bad_rows = int(non_numeric.sum())
        if bad_rows > 0:
            errors.append(
                f"Column '{col}' contains {bad_rows} non-numeric values"
            )

    if "TotalCharges" in actual_columns:
        blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
        blank_count = int(blank_mask.sum())
        if blank_count > 0:
            warnings.append(
                f"Column 'TotalCharges' has {blank_count} blank/whitespace-only values"
            )

    if "SeniorCitizen" in actual_columns:
        senior_values = set(pd.to_numeric(df["SeniorCitizen"], errors="coerce").dropna().unique())
        invalid_senior = senior_values - {0.0, 1.0}
        if invalid_senior:
            errors.append(
                f"Column 'SeniorCitizen' has values outside {{0, 1}}: {sorted(invalid_senior)}"
            )

    report = {
        "file": input_path,
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "duplicates": duplicate_count,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    if errors:
        logger.error("Validation FAILED with %d error(s)", len(errors))
        for e in errors:
            logger.error("  ERROR: %s", e)
    else:
        logger.info("Validation PASSED")

    if warnings:
        for w in warnings:
            logger.warning("  WARN: %s", w)

    report_json = json.dumps(report, indent=2)
    logger.info("Validation report:\n%s", report_json)

    return report_json


if __name__ == "__main__":
    validate_data(input_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
