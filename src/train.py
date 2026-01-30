import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = os.getenv("DATA_PATH", "data/sample.csv")

def main():
    df = pd.read_csv(DATA_PATH)

    X = df[["age", "income"]]
    y = df["clicked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    # MLflow only runs in CD
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.start_run()
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()

if __name__ == "__main__":
    main()
