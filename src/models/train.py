import pandas as pd
from src.config import PROCESSED_DATA_PATH
from src.data.make_dataset import build_modeling_dataset
from src.pipelines.training_pipeline import train_and_compare_models

def main():
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["transactiondate"])
    else:
        df = build_modeling_dataset()
    results_df, best_name, best_metrics = train_and_compare_models(df)
    print("\nModel comparison:")
    print(results_df)
    print(f"\nBest model: {best_name}")
    print(f"Best metrics: {best_metrics}")

if __name__ == "__main__":
    main()
