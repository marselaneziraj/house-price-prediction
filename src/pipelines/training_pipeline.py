import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.config import BEST_MODEL_METRICS_PATH, BEST_MODEL_PATH, FEATURE_IMPORTANCE_PATH, MODEL_RESULTS_PATH, RANDOM_STATE
from src.features.preprocess import select_features, build_preprocessor
from src.models.metrics import regression_metrics

def split_modeling_data(df: pd.DataFrame, test_size: float = 0.2):
    X, y, numeric_cols, categorical_cols = select_features(df)
    if X.empty:
        raise ValueError("No features are available after preprocessing selection.")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def train_and_compare_models(df: pd.DataFrame):
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = split_modeling_data(df)
    model_candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
    }
    results = []
    trained = {}
    for model_name, model in model_candidates.items():
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        metrics["model"] = model_name
        results.append(metrics)
        trained[model_name] = pipeline
        print(model_name, metrics)
    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    results_df.to_csv(MODEL_RESULTS_PATH, index=False)
    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    best_metrics = results_df.iloc[0].to_dict()
    joblib.dump(best_model, BEST_MODEL_PATH)
    with open(BEST_MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    model_obj = best_model.named_steps["model"]
    preprocessor_obj = best_model.named_steps["preprocessor"]
    FEATURE_IMPORTANCE_PATH.unlink(missing_ok=True)
    if hasattr(model_obj, "feature_importances_"):
        feature_names = preprocessor_obj.get_feature_names_out()
        feature_importance = pd.DataFrame({"feature": feature_names, "importance": model_obj.feature_importances_}).sort_values(by="importance", ascending=False)
        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return results_df, best_name, best_metrics
