import json
import joblib
import pandas as pd
from src.config import BEST_MODEL_PATH, PROCESSED_DATA_PATH
from src.data.make_dataset import build_modeling_dataset
from src.models.metrics import regression_metrics
from src.pipelines.training_pipeline import split_modeling_data

def main():
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["transactiondate"])
    else:
        df = build_modeling_dataset()
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {BEST_MODEL_PATH}. Run `python -m src.models.train` first."
        )
    _, X_test, _, y_test, _, _ = split_modeling_data(df)
    model = joblib.load(BEST_MODEL_PATH)
    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
