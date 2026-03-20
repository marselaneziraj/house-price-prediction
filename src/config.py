from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

TRAIN_2016_PATH = RAW_DATA_DIR / "train_2016_v2.csv"
TRAIN_2017_PATH = RAW_DATA_DIR / "train_2017.csv"
PROPERTIES_2016_PATH = RAW_DATA_DIR / "properties_2016.csv"
PROPERTIES_2017_PATH = RAW_DATA_DIR / "properties_2017.csv"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "modeling_dataset.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_RESULTS_PATH = MODELS_DIR / "model_results.csv"
BEST_MODEL_METRICS_PATH = MODELS_DIR / "best_model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

RANDOM_STATE = 42

for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
