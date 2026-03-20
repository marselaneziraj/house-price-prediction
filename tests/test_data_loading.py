import pandas as pd
import pytest

from src.data.load_data import merge_year_data, validate_raw_data_files


def test_validate_raw_data_files_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_raw_data_files(tmp_path)

    message = str(exc_info.value)
    assert "train_2016_v2.csv" in message
    assert "properties_2017.csv" in message
    assert str(tmp_path) in message


def test_merge_year_data_rejects_duplicate_property_rows():
    train_df = pd.DataFrame(
        {
            "parcelid": [1, 2],
            "data_year": [2016, 2016],
            "logerror": [0.1, -0.2],
        }
    )
    properties_df = pd.DataFrame(
        {
            "parcelid": [1, 1],
            "data_year": [2016, 2016],
            "yearbuilt": [1990, 1991],
        }
    )

    with pytest.raises(pd.errors.MergeError):
        merge_year_data(train_df, properties_df)
