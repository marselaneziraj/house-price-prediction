import numpy as np
import pandas as pd

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "transactiondate" not in df.columns:
        return df
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    return df

def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"yearbuilt", "transaction_year"}.issubset(df.columns):
        df["property_age"] = df["transaction_year"] - df["yearbuilt"]
    if {"bathroomcnt", "bedroomcnt"}.issubset(df.columns):
        df["bath_bed_ratio"] = df["bathroomcnt"] / (df["bedroomcnt"].replace(0, np.nan))
    if {"calculatedfinishedsquarefeet", "roomcnt"}.issubset(df.columns):
        df["avg_room_size"] = df["calculatedfinishedsquarefeet"] / (df["roomcnt"].replace(0, np.nan))
    if {"structuretaxvaluedollarcnt", "taxvaluedollarcnt"}.issubset(df.columns):
        df["structure_tax_ratio"] = df["structuretaxvaluedollarcnt"] / (df["taxvaluedollarcnt"].replace(0, np.nan))
    if {"landtaxvaluedollarcnt", "taxvaluedollarcnt"}.issubset(df.columns):
        df["land_tax_ratio"] = df["landtaxvaluedollarcnt"] / (df["taxvaluedollarcnt"].replace(0, np.nan))
    if {"taxamount", "taxvaluedollarcnt"}.issubset(df.columns):
        df["effective_tax_rate"] = df["taxamount"] / (df["taxvaluedollarcnt"].replace(0, np.nan))
    return df

def clean_basic_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hashottuborspa" in df.columns:
        df["hashottuborspa"] = df["hashottuborspa"].astype("string").fillna("False")
    if "fireplaceflag" in df.columns:
        df["fireplaceflag"] = df["fireplaceflag"].astype("string").fillna("False")
    return df


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_property_features(df)
    return clean_basic_values(df)
