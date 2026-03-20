from pathlib import Path

import pandas as pd
from src.config import (
    RAW_DATA_DIR,
    TRAIN_2016_PATH,
    TRAIN_2017_PATH,
    PROPERTIES_2016_PATH,
    PROPERTIES_2017_PATH,
)

REQUIRED_RAW_FILENAMES = (
    "train_2016_v2.csv",
    "train_2017.csv",
    "properties_2016.csv",
    "properties_2017.csv",
)


def validate_raw_data_files(raw_data_dir: Path | None = None) -> None:
    base_dir = Path(raw_data_dir) if raw_data_dir is not None else RAW_DATA_DIR
    missing_files = [base_dir / name for name in REQUIRED_RAW_FILENAMES if not (base_dir / name).exists()]
    if missing_files:
        missing_list = "\n".join(f"- {path.name}" for path in missing_files)
        raise FileNotFoundError(
            "Missing required raw data files.\n"
            f"Searched in: {base_dir}\n"
            "Add the following files before building the dataset:\n"
            f"{missing_list}"
        )


def load_train_data():
    train_2016 = pd.read_csv(TRAIN_2016_PATH, parse_dates=["transactiondate"])
    train_2017 = pd.read_csv(TRAIN_2017_PATH, parse_dates=["transactiondate"])
    train_2016["data_year"] = 2016
    train_2017["data_year"] = 2017
    return train_2016, train_2017


def load_property_data():
    properties_2016 = pd.read_csv(PROPERTIES_2016_PATH, low_memory=False)
    properties_2017 = pd.read_csv(PROPERTIES_2017_PATH, low_memory=False)
    properties_2016["data_year"] = 2016
    properties_2017["data_year"] = 2017
    return properties_2016, properties_2017


def merge_year_data(train_df, properties_df):
    return train_df.merge(
        properties_df,
        on=["parcelid", "data_year"],
        how="left",
        validate="many_to_one",
    )


def load_and_merge_all_years():
    validate_raw_data_files()
    train_2016, train_2017 = load_train_data()
    properties_2016, properties_2017 = load_property_data()
    merged_2016 = merge_year_data(train_2016, properties_2016)
    merged_2017 = merge_year_data(train_2017, properties_2017)
    return pd.concat([merged_2016, merged_2017], axis=0, ignore_index=True)
