from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "scikit-learn"],
)
def feature_engineering(input_path: str, output_dir: str = "data/features",) -> str:
    """
    Prepare features for model training
    Engineer new features:
         - tenure_group: bucketed tenure (0-12, 13-24, 25-48, 49-60, 61-72)
         - avg_monthly_charge: TotalCharges / tenure
         - service_count: number of add-on services the customer subscribes to
         - monthly_charge_x_tenure: interaction between monthly spend and loyalty

    Returns the path to the training data directory
    """
    import json
    import logging
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    TARGET = "Churn"

    SERVICE_COLUMNS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df = pd.read_csv(input_path)
    logger.info("Loaded dataset: %s", df.shape)

    tenure_bins = [0, 12, 24, 48, 60, 72]
    tenure_labels = [0, 1, 2, 3, 4]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
    ).astype(int)
    logger.info("Created tenure_group")

    df["avg_monthly_charge"] = df.apply(
        lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else 0.0,
        axis=1,
    )
    logger.info("Created avg_monthly_charge")

    active_services = [c for c in SERVICE_COLUMNS if c in df.columns]
    df["service_count"] = df[active_services].sum(axis=1)
    logger.info("Created service_count from %d service columns", len(active_services))

    df["monthly_charge_x_tenure"] = df["MonthlyCharges"] * df["tenure"]
    logger.info("Created monthly_charge_x_tenure")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(
        "Split -> train: %s  val: %s", X_train.shape, X_val.shape,
    )

    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    logger.info("Scaled %d numeric columns", len(numeric_cols))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    train_path = output_path / "train.csv"
    val_path = output_path / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    scaler_params = {
        "numeric_columns": numeric_cols,
        "means": {col: float(scaler.mean_[i]) for i, col in enumerate(numeric_cols)},
        "stds": {col: float(scaler.scale_[i]) for i, col in enumerate(numeric_cols)},
    }
    scaler_path = output_path / "scaler_params.json"
    scaler_path.write_text(json.dumps(scaler_params, indent=2))

    logger.info("Saved train  -> %s", train_path)
    logger.info("Saved val    -> %s", val_path)
    logger.info("Saved scaler -> %s", scaler_path)
    logger.info("Feature columns: %s", X_train.columns.tolist())

    return str(output_path)


if __name__ == "__main__":
    feature_engineering(input_path="data/cleaned/cleaned.csv")
