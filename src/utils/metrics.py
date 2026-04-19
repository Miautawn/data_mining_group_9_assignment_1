import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.utils.data import decode_labels


def calculate_regression_metrics(
    y_true: np.array,
    y_pred: np.array,
    user_ids: np.array,
    preprocessor: ColumnTransformer,
) -> dict:
    """
    Calculates Micro and Macro evaluation metrics (MSE, RMSE, MAE).
    Also decodes the labels and predictions back to the original mood points scale before calculating the metrics.

    Args:
        y_true: Array-like of decoded actual labels
        y_pred: Array-like of decoded predictions
        user_ids: Array-like of user identifiers
    """

    y_true = decode_labels(preprocessor, user_ids, y_true)
    y_pred = decode_labels(preprocessor, user_ids, y_pred)

    df = pd.DataFrame({"user_id": user_ids, "true": y_true, "pred": y_pred})

    # 1. Micro Metrics (Global)
    micro_mse = mean_squared_error(y_true, y_pred)
    micro_rmse = np.sqrt(micro_mse)
    micro_mae = mean_absolute_error(y_true, y_pred)

    # 2. Macro Metrics (Per-User average)
    user_metrics = df.groupby("user_id").apply(
        lambda g: pd.Series(
            {
                "mse": mean_squared_error(g["true"], g["pred"]),
                "mae": mean_absolute_error(g["true"], g["pred"]),
            }
        ),
        include_groups=False,
    )

    macro_mse = user_metrics["mse"].mean()
    macro_rmse = np.sqrt(user_metrics["mse"]).mean()
    macro_mae = user_metrics["mae"].mean()

    # Format the results
    metrics = {
        "Micro": {
            "MSE": round(micro_mse, 4),
            "RMSE": round(micro_rmse, 4),
            "MAE": round(micro_mae, 4),
        },
        "Macro": {
            "MSE": round(macro_mse, 4),
            "RMSE": round(macro_rmse, 4),
            "MAE": round(macro_mae, 4),
        },
    }

    return metrics


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculates key multiclass classification metrics.

    Args:
        y_true: 1D array-like of actual class labels
        y_pred: 1D array-like of predicted class labels
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    metrics = {
        "Macro": {"F1": round(macro_f1, 4)},
        "Micro": {"F1": round(micro_f1, 4)},
    }

    return metrics
