import mlflow
import os
import sys

# Connect to MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

# Read the Run ID from model_info.txt
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking accuracy for Run ID: {run_id}")

# Get the run details from MLflow
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Accuracy found: {accuracy}")

# Check if accuracy meets the threshold
if accuracy < 0.85:
    print(f"FAILED! Accuracy {accuracy} is below threshold 0.85")
    sys.exit(1)
else:
    print(f"PASSED! Accuracy {accuracy} is above threshold 0.85")
    sys.exit(0)