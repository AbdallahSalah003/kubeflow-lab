from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def train_model(
    train_dataset: Input[Dataset],
    model: Output[Model],
):
    """
    Train a Logistic Regression model
    for Telco Customer Churn classification.
    """
    import json
    import logging
    from pathlib import Path

    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    TARGET = "Churn"

    train_df = pd.read_csv(train_dataset.path)

    X_train = train_df.drop(columns=[TARGET])

    y_train = train_df[TARGET]

    logger.info(
        "Training data: %s",
        X_train.shape,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    logger.info("Logistic Regression trained successfully")

    model_path = Path(model.path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        classifier,
        model_path,
    )

    model.metadata["framework"] = "scikit-learn"
    model.metadata["model_type"] = "LogisticRegression"
    model.metadata["features"] = X_train.shape[1]

    logger.info(
        "Model saved to: %s",
        model_path,
    )
