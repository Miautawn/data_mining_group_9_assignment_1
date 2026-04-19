from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.utils.data import decode_labels


def baseline_regression(X: pd.DataFrame):
    """Simply returns the last target value from the window.
    The idea is that mood is autocorrelated, so the last mood value is probably correct

    Args:
        X (pd.DataFrame): input dataframe

    Returns:
        tuple(np.array, np.array, np.array): true labels, predictions, user ids
    """
    y_true = []
    y_pred = []
    user_ids = []

    for name, group in X.groupby("window_id"):
        y_true.append(group["target_mean_mood"].iloc[0])
        user_ids.append(group["id"].iloc[0])
        y_pred.append(group.iloc[-1]["mood_mean"])

    return np.array(y_true), np.array(y_pred), np.array(user_ids)


def baseline_classification(X: pd.DataFrame, mood_to_class: Callable):
    """Simply returns the last target value from the window.
    The idea is that mood is autocorrelated, so the last mood value is probably correct

    Args:
        X (pd.DataFrame): input dataframe

    Returns:
        tuple(np.array, np.array, np.array): true labels, predictions, user ids
    """
    y_true = []
    y_pred = []

    for name, group in X.groupby("window_id"):
        y_true.append(group["mood_label"].iloc[0])
        y_pred.append(mood_to_class(group.iloc[-1]["mood_mean"]))

    return np.array(y_true), np.array(y_pred)
