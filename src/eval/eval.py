from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def evaluate_model(
    model: Input[Model],
    validation_dataset: Input[Dataset],
    metrics: Output[Metrics],
):
    """
    Evaluate the trained churn classification model.
    """
    import logging

    import joblib
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger(__name__)

    TARGET = "Churn"

    classifier = joblib.load(model.path)

    val_df = pd.read_csv(validation_dataset.path)

    X_val = val_df.drop(columns=[TARGET])

    y_val = val_df[TARGET]

    logger.info(
        "Validation dataset: %s",
        X_val.shape,
    )

    y_pred = classifier.predict(X_val)

    y_prob = classifier.predict_proba(X_val)[:, 1]

    accuracy = float(
        accuracy_score(
            y_val,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_val,
            y_prob,
        )
    )

    logger.info(
        "Accuracy: %.4f",
        accuracy,
    )

    logger.info(
        "Precision: %.4f",
        precision,
    )

    logger.info(
        "Recall: %.4f",
        recall,
    )

    logger.info(
        "F1: %.4f",
        f1,
    )

    logger.info(
        "ROC-AUC: %.4f",
        roc_auc,
    )

    metrics.log_metric(
        "accuracy",
        accuracy,
    )

    metrics.log_metric(
        "precision",
        precision,
    )

    metrics.log_metric(
        "recall",
        recall,
    )

    metrics.log_metric(
        "f1",
        f1,
    )

    metrics.log_metric(
        "roc_auc",
        roc_auc,
    )
