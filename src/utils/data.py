from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_windows(
    dataset: Optional[pd.DataFrame] = None,
    dataset_target_splits: Optional[pd.DataFrame] = None,
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Takes in the source dataset together with the dataset target day splits to produce
    train/val/test dataset windows comprised of original events (tall format).

    This function simply outputs the `window_size` day windows for each mood target
        but does not perform any preprocessing (e.g. dataset reindexing for missing day gaps) or feature engineering.

    You are supposed to take the outputs of this function and turn into actual data points for your model pipeline yourself.

    Args:
        dataset (pd.DataFrame): source dataset (tall format).
            If not provided, will take from the default path `data/1b_dataset_cleaned.parquet`.
        dataset_target_splits (pd.DataFrame): dataset target day splits (id, date, split, target_mean_mood)
            If not provided, will take from the default path `data/1c_dataset_target_splits.parquet`.
        window_size (int, optional): size of the input window prior to the target prediction day.
            For example, if one target from the dataset_target_splits is on `2026-01-08` and the window_size is 7,
            then the input window will be from `2026-01-01` to `2026-01-07` (inclusive). Defaults to 7.

    Returns:
        pd.DataFrame: dataset windows comprised of original events (tall format)
            with the inication whether they belong to train/val/test split.
    """

    if dataset is None:
        dataset = pd.read_parquet(DATA_DIR / "1b_dataset_cleaned.parquet")
    else:
        dataset = dataset.copy()

    if dataset_target_splits is None:
        dataset_target_splits = pd.read_parquet(
            DATA_DIR / "1b_dataset_target_splits.parquet"
        )
    else:
        dataset_target_splits = dataset_target_splits.copy()

    dataset["date"] = dataset["time"].dt.normalize()
    dataset_target_splits.rename(columns={"date_target": "date"}, inplace=True)
    dataset_target_splits["start_date"] = dataset_target_splits["date"] - pd.Timedelta(
        days=window_size
    )

    # Join dataset to target_splits on 'id'
    # This performs a cartesian product for each user
    # which we then will filter down to relevant windows in a single pass
    merged = pd.merge(dataset, dataset_target_splits, on="id", suffixes=("", "_target"))

    mask = (merged["date"] >= merged["start_date"]) & (
        merged["date"] < merged["date_target"]
    )
    final_df = merged[mask]
    final_df["window_id"] = (
        final_df["split"]
        + "_"
        + final_df["id"]
        + "_"
        + final_df["date_target"].astype(str)
    )
    final_df["window_size_days"] = window_size

    final_df = final_df[
        [
            "id",
            "time",
            "date",
            "variable",
            "value",
            "split",
            "date_target",
            "target_mean_mood",
            "window_id",
            "window_size_days",
        ]
    ]

    return final_df


class UserStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardizes features per user. Fits on training data to learn
    per-user means/stds and applies them to transform data.
    """

    def __init__(self, id_col: str = "id"):
        self.id_col = id_col
        self.columns_to_standardize = []
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_to_standardize = [col for col in X.columns if col != self.id_col]

        # We use a dictionary for fast lookup: {user_id: {col: (mean, std)}}
        grouped = X.groupby(self.id_col)
        for user_id, group in grouped:
            user_dict = {}
            for col in self.columns_to_standardize:
                vals = group[col]
                user_dict[col] = (np.mean(vals), np.std(vals) + 1e-6)
            self.stats_[user_id] = user_dict

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["stats_"])
        X_out = X.copy()

        for col in self.columns_to_standardize:
            stats = X_out[self.id_col].map(lambda uid: self.stats_.get(uid).get(col))
            means, stds = zip(*stats)

            X_out[col] = (X_out[col] - means) / stds

        return X_out[self.columns_to_standardize]

    def get_feature_names_out(self, input_features=None):
        return self.columns_to_standardize


class LocalUserScaler(BaseEstimator, TransformerMixin):
    """
    Standardizes features per user.
    This is an amalgamation that shouldn't have seen the light of day!

    Because we're splitting the data by users, we can't just fit on users in the train
    as we wouldn't be able to descale the val/test users.

    So instead, the user means/std are saved on each transform call.
    """

    def __init__(self, id_col="id", cols=None):
        self.id_col = id_col
        self.cols = cols
        self.stats_ = {}

    def fit(self, X, y=None):
        self.cols = [col for col in X.columns if col != self.id_col]
        return self  # Nothing to "learn" globally

    def transform(self, X):
        X_out = X.copy()
        # Group by the user ID present in THIS specific dataframe (Train, Val, or Test)
        for user_id, grouped in X_out.groupby(self.id_col):
            user_dict = {}
            for col in self.cols:
                vals = grouped[col]
                user_dict[col] = (np.mean(vals), np.std(vals) + 1e-6)

            self.stats_[user_id] = user_dict

        for col in self.cols:
            stats = X_out[self.id_col].map(lambda uid: self.stats_.get(uid).get(col))
            means, stds = zip(*stats)

            X_out[col] = (X_out[col] - means) / stds

        return X_out[self.cols]

    def get_feature_names_out(self, input_features=None):
        return self.cols
