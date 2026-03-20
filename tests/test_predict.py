import numpy as np
import pandas as pd

from src.models.predict import align_features_to_model_input


class DummyModel:
    feature_names_in_ = np.array(["property_age", "missing_numeric"])


def test_align_features_to_model_input_adds_missing_expected_columns():
    df = pd.DataFrame(
        {
            "transactiondate": ["2017-02-01"],
            "yearbuilt": [2000],
        }
    )

    aligned = align_features_to_model_input(df, DummyModel())

    assert list(aligned.columns) == ["property_age", "missing_numeric"]
    assert aligned.loc[0, "property_age"] == 17
    assert pd.isna(aligned.loc[0, "missing_numeric"])
