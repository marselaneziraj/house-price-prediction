import pandas as pd

from src.features.feature_engineering import add_property_features, prepare_feature_frame


def test_add_property_features_handles_zero_denominators():
    df = pd.DataFrame(
        {
            "transaction_year": [2017],
            "yearbuilt": [2000],
            "bathroomcnt": [2.0],
            "bedroomcnt": [0.0],
            "calculatedfinishedsquarefeet": [1200.0],
            "roomcnt": [0.0],
            "structuretaxvaluedollarcnt": [150000.0],
            "taxvaluedollarcnt": [0.0],
        }
    )

    result = add_property_features(df)

    assert result.loc[0, "property_age"] == 17
    assert pd.isna(result.loc[0, "bath_bed_ratio"])
    assert pd.isna(result.loc[0, "avg_room_size"])
    assert pd.isna(result.loc[0, "structure_tax_ratio"])


def test_prepare_feature_frame_adds_date_features_and_normalizes_flags():
    df = pd.DataFrame(
        {
            "transactiondate": ["2017-01-15"],
            "yearbuilt": [2000],
            "hashottuborspa": [None],
            "fireplaceflag": [True],
        }
    )

    result = prepare_feature_frame(df)

    assert result.loc[0, "transaction_year"] == 2017
    assert result.loc[0, "transaction_month"] == 1
    assert result.loc[0, "transaction_quarter"] == 1
    assert result.loc[0, "property_age"] == 17
    assert result.loc[0, "hashottuborspa"] == "False"
    assert result.loc[0, "fireplaceflag"] == "True"
