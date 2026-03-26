import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("assignment5")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy}")
    print(f"Run ID: {run.info.run_id}")

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

print("Training complete! model_info.txt created.")