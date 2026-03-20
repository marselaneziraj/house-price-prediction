from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def select_features(df: pd.DataFrame, target_col: str = "logerror", missing_threshold: float = 0.70, max_categorical_cardinality: int = 30):
    df = df.copy()
    protected_cols = {target_col, "parcelid", "transactiondate"}
    candidate_cols = [c for c in df.columns if c not in protected_cols]
    missing_ratio = df[candidate_cols].isna().mean()
    kept_cols = [c for c in candidate_cols if missing_ratio[c] <= missing_threshold]
    X = df[kept_cols].copy()
    y = df[target_col].copy()
    categorical_cols = []
    numeric_cols = []
    for col in X.columns:
        if X[col].dtype == "object":
            nunique = X[col].nunique(dropna=True)
            if nunique <= max_categorical_cardinality:
                categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    X = X[numeric_cols + categorical_cols].copy()
    return X, y, numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols), ("cat", categorical_transformer, categorical_cols)], remainder="drop")
