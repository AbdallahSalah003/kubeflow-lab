from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
    ],
)
def feature_engineering(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    validation_dataset: Output[Dataset],
):
    """
    Feature engineering and train/validation split.
    """
    import logging

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

    df = pd.read_csv(input_dataset.path)

    logger.info(
        "Loaded dataset: %s",
        df.shape,
    )

    # -------------------------
    # Feature engineering
    # -------------------------

    tenure_bins = [
        0,
        12,
        24,
        48,
        60,
        72,
    ]

    tenure_labels = [
        0,
        1,
        2,
        3,
        4,
    ]

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
    ).astype(int)

    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    active_services = [column for column in SERVICE_COLUMNS if column in df.columns]

    df["service_count"] = df[active_services].sum(axis=1)

    df["monthly_charge_x_tenure"] = df["MonthlyCharges"] * df["tenure"]

    # -------------------------
    # Split
    # -------------------------

    y = df[TARGET]

    X = df.drop(columns=[TARGET])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(
        "Train: %s | Validation: %s",
        X_train.shape,
        X_val.shape,
    )

    # -------------------------
    # Scaling
    # -------------------------

    scaler = StandardScaler()

    numeric_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

    X_val[numeric_columns] = scaler.transform(X_val[numeric_columns])

    # -------------------------
    # Save
    # -------------------------

    train_df = pd.concat(
        [X_train, y_train],
        axis=1,
    )

    val_df = pd.concat(
        [X_val, y_val],
        axis=1,
    )

    train_df.to_csv(
        train_dataset.path,
        index=False,
    )

    validation_dataset_path = validation_dataset.path

    val_df.to_csv(
        validation_dataset_path,
        index=False,
    )

    logger.info(
        "Training dataset saved to: %s",
        train_dataset.path,
    )

    logger.info(
        "Validation dataset saved to: %s",
        validation_dataset_path,
    )
