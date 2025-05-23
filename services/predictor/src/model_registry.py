import mlflow
from loguru import logger
import pandas as pd
from mlflow.models.signature import infer_signature

def load_mode_from_registry(
        model_name: str,
        model_version: str,
        mlflow_tracking_uri: str
)   :

    # Set the tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(model_uri=f'models:/{model_name}/{model_version}')

    # Get model info
    model_info = mlflow.models.get_model_info(model_uri=f'models:/{model_name}/{model_version}')
    features = model_info.signature.inputs.input_names()
    
    return model, features

def push_model_to_registry(
        best_model,
        experiment_name: str,
        X_train: pd.DataFrame,
        ratio_diff_mae: float
    ):

    signature = infer_signature(X_train, best_model.predict(X_train))

    # Load model in the registry model mlflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        registered_model_name=experiment_name,
    )
    logger.info(
        f"New model registered. Ratio difference with base model: {ratio_diff_mae}"
    )