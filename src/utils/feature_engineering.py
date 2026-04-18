import numpy as np
import pandas as pd


def compute_daily_aggregates(df: pd.DataFrame, agg_rules: dict) -> pd.DataFrame:
    """Compute daily aggregates for the given DataFrame.
    Will be used to compute the features for the RNN model.
    Each row will contain a daily aggregates for all variable types (mood, activity, calls, etc.) for a specific window and date.

    Args:
        df (pd.DataFrame): dataframe containing windowed events data (tall format).
            E.g. output from the `src.utils.data.get_windows` function

        agg_rules (dict): aggregation rules specifying which statistics to compute for each variable.
            E.g. {
                "mood": ["count", "meanstd"],
                "circumplex.arousal": ["count", "mean", "std"],
                ...

    Returns:
        pd.DataFrame: dataframe containing:
        - daily aggregates for each variable and window.
        - cyclically encoded date features (day_sin, day_cos).
        - mean mood for the target date of the window (target_mean_mood).
    """

    df = df.copy()

    # compute daily aggregates
    daily_agg = df.groupby(["window_id", "date", "variable"])["value"].agg(
        ["count", "mean", "sum", "std"]
    )
    wide_df = daily_agg.unstack("variable")

    # filter out only the aggregates we need
    valid_columns = []
    for stat, var in wide_df.columns:
        if var in agg_rules and stat in agg_rules[var]:
            valid_columns.append((stat, var))

    wide_df = wide_df[valid_columns]

    # Flatten names (e.g., 'mood_mean', 'sms_sum')
    wide_df.columns = [f"{var}_{stat}" for stat, var in wide_df.columns]
    wide_df = wide_df.reset_index()

    # ensure the windows have the same sequence lengths)
    window_skeleton = _get_window_skeleton(df)
    complete_wide_df = window_skeleton.merge(
        wide_df, on=["window_id", "date"], how="left"
    )

    # Add the missingness indicators
    for col in agg_rules.keys():
        count_key = f"{col}_count"
        complete_wide_df[f"{col}_mask"] = complete_wide_df[count_key].isna().astype(int)

    # impute missing values
    complete_wide_df = complete_wide_df.fillna(0)

    # add the cyclical date features
    day_of_week = complete_wide_df["date"].dt.dayofweek
    complete_wide_df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    complete_wide_df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    return complete_wide_df


def _get_window_skeleton(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a skeleton DataFrame that fills each day in the window.
        for each date in the window based on the `window_size_days` column in the input dataframe.
        (note: it ensures that all dates would be present even if there are no readings in the source dataframe)

    This dataframe should be used to join feature aggregates to ensure that the windows will be
    of the same sequence length.

    Args:
        df (pd.DataFrame): The input dataframe containing window information.

    Returns:
        pd.DataFrame: A skeleton dataframe with the following columns:
            - window_id: Unique identifier for each window.
            - id: ID of the patient.
            - split: The data split (e.g., train, test, val).
            - date: Each date in the window based on the `window_size_days` column in the input dataframe
            - target_mean_mood: The mean mood for the target date of the window.
    """
    window_size = df["window_size_days"].iloc[0]
    unique_windows = df.drop_duplicates(subset=["window_id"])[
        ["window_id", "id", "split", "date_target", "target_mean_mood"]
    ]

    offsets = pd.DataFrame({"day_offset": range(1, window_size + 1)})

    skeleton = unique_windows.merge(offsets, how="cross")

    skeleton["date"] = pd.to_datetime(skeleton["date_target"]) - pd.to_timedelta(
        skeleton["day_offset"], unit="D"
    )

    return skeleton[["window_id", "id", "split", "date", "target_mean_mood"]]
