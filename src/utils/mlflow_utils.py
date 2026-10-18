
import mlflow
from pathlib import Path

def setup_mlflow(experiment_name: str = "retail-demand-forecasting") -> None:
    """Configure local MLflow experiment tracking."""

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
