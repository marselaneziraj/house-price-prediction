import pandas as pd

from src.features.preprocess import select_features


def test_select_features_filters_missing_and_high_cardinality_columns():
    df = pd.DataFrame(
        {
            "parcelid": [1, 2, 3],
            "transactiondate": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "logerror": [0.1, -0.1, 0.05],
            "numeric_feature": [10.0, 20.0, 30.0],
            "mostly_missing": [None, None, 1.0],
            "small_category": ["A", "B", "A"],
            "high_cardinality": ["id1", "id2", "id3"],
        }
    )

    X, y, numeric_cols, categorical_cols = select_features(
        df,
        missing_threshold=0.5,
        max_categorical_cardinality=2,
    )

    assert list(X.columns) == ["numeric_feature", "small_category"]
    assert numeric_cols == ["numeric_feature"]
    assert categorical_cols == ["small_category"]
    assert y.tolist() == [0.1, -0.1, 0.05]
