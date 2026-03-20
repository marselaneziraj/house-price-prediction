import numpy as np
import joblib
import pandas as pd
from src.config import BEST_MODEL_PATH
from src.features.feature_engineering import prepare_feature_frame

def load_trained_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {BEST_MODEL_PATH}. Run `python -m src.models.train` first."
        )
    return joblib.load(BEST_MODEL_PATH)


def align_features_to_model_input(df: pd.DataFrame, model) -> pd.DataFrame:
    prepared_df = prepare_feature_frame(df)
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is None:
        return prepared_df
    aligned_df = prepared_df.copy()
    missing_features = [feature for feature in expected_features if feature not in aligned_df.columns]
    for feature in missing_features:
        aligned_df[feature] = np.nan
    return aligned_df.loc[:, list(expected_features)]

def predict_from_csv(input_csv_path: str, output_csv_path: str):
    model = load_trained_model()
    df = pd.read_csv(input_csv_path)
    model_input = align_features_to_model_input(df, model)
    predictions = model.predict(model_input)
    output = df.copy()
    output["predicted_logerror"] = predictions
    output.to_csv(output_csv_path, index=False)
    return output

if __name__ == "__main__":
    raise SystemExit("Use predict_from_csv(input_csv_path, output_csv_path) from another script or notebook.")
