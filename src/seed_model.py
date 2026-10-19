import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import time

mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)

model_path = "/app/models/champion_model_cache.joblib"

print("Waiting")
time.sleep(10)

client = MlflowClient()

try:
    client.get_registered_model("olist_model")
    print("Model already exists in MLflow")
except Exception as e:
    print("Model not found in MLflow. Attempting to seed from local cache")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        mlflow.set_experiment("olist_production")
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="olist_model",
                serialization_format="cloudpickle",
                code_paths=["src/"]
            )
            
        latest_version = client.get_latest_versions("olist_model", stages=["None"])[0].version
        client.set_registered_model_alias("olist_model", "champion", latest_version)
        print(f"Model seeded and aliased as champion (Version {latest_version}).")
    else:
        print(f"File {model_path} not found. Cannot seed MLflow.")