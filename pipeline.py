from kfp import dsl

from src.data.collection import collect_data
from src.data.validation import validate_data
from src.data.cleaning import clean_data
from src.features.features import feature_engineering
from src.train.train import train_model
from src.eval.eval import evaluate_model


@dsl.pipeline(
    name="telco-churn-pipeline",
)
def telco_churn_pipeline():

    collection_task = collect_data()


    validation_task = validate_data(
        input_dataset=collection_task.outputs[
            "output_dataset"
        ],
    )


    cleaning_task = clean_data(
        input_dataset=collection_task.outputs[
            "output_dataset"
        ],

        validation_report=validation_task.outputs[
            "validation_report"
        ],
    )


    feature_task = feature_engineering(
        input_dataset=cleaning_task.outputs[
            "output_dataset"
        ],
    )


    training_task = train_model(
        train_dataset=feature_task.outputs[
            "train_dataset"
        ],
    )


    evaluate_task = evaluate_model(
        model=training_task.outputs[
            "model"
        ],

        validation_dataset=feature_task.outputs[
            "validation_dataset"
        ],
    )


if __name__ == "__main__":

    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=telco_churn_pipeline,
        package_path="telco_churn_pipeline.yaml",
    )

    print(
        "Pipeline compiled to "
        "telco_churn_pipeline.yaml"
    )