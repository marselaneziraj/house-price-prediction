import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, PROCESSED_DATA_PATH, RANDOM_STATE
from src.data.make_dataset import build_modeling_dataset
from src.features.preprocess import select_features, build_preprocessor
from src.models.metrics import regression_metrics


def main():
    # Load processed data if available, otherwise build it
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["transactiondate"])
    else:
        df = build_modeling_dataset()

    # Select features and target
    X, y, numeric_cols, categorical_cols = select_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Use a smaller sample for tuning to reduce runtime
    if len(X_train) > 20000:
        X_train = X_train.sample(20000, random_state=RANDOM_STATE)
        y_train = y_train.loc[X_train.index]

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Base model
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ])

    # Smaller search space for faster tuning
    param_distributions = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [10, 15, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.5],
    }

    # Randomized search
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="neg_root_mean_squared_error",
        cv=2,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train search
    search.fit(X_train, y_train)

    # Evaluate best model on test set
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    metrics = regression_metrics(y_test, preds)

    # Save results
    output = {
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "metrics": metrics
    }

    out_path = MODELS_DIR / "random_forest_tuning_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()