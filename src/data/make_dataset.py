import pandas as pd
from src.config import PROCESSED_DATA_PATH
from src.data.load_data import load_and_merge_all_years
from src.features.feature_engineering import prepare_feature_frame

def build_modeling_dataset() -> pd.DataFrame:
    df = load_and_merge_all_years()
    df = df.drop_duplicates().copy()
    df = prepare_feature_frame(df)
    if "logerror" not in df.columns:
        raise KeyError("The merged Zillow dataset must include a 'logerror' target column.")
    df = df[df["logerror"].notna()].copy()
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset to: {PROCESSED_DATA_PATH}")
    print(f"Processed shape: {df.shape}")
    return df

if __name__ == "__main__":
    build_modeling_dataset()
