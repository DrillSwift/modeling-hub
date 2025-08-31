import pandas as pd
import numpy as np

def shifted_labels(
    df: pd.DataFrame,
    segmented_end_time_array: np.ndarray = None,
    shift_minutes: int = 15,
    label_col: str = "Category",
    sort_by: str = "TIME",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate shifted labels for early prediction.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'TIME' column in datetime format.
        segmented_end_time_array (np.ndarray, optional): Array of end times for windows. Defaults to None.
        shift_minutes (int, optional): Minimum time shift for early prediction in minutes. Defaults to 15.
        label_col (str, optional): Name of the label column. Defaults to 'Category'.
        sort_by (str, optional): Column to sort the DataFrame by. Defaults to 'TIME'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Shifted labels array.
            - Shifted label times array.

    Raises:
        ValueError: If 'TIME' or label_col is not in DataFrame, or if no valid windows are found.
        ValueError: If segmented_end_time_array contains non-datetime values.
    """
    if df.empty:
        return np.array([]), np.array([], dtype="datetime64[ns]")
    if sort_by not in df.columns:
        raise ValueError(f"DataFrame must contain a '{sort_by}' column.")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    # Sort DataFrame if sort_by column exists
    df = df.sort_values(sort_by).reset_index(drop=True)

    # Convert TIME to datetime
    try:
        df[sort_by] = pd.to_datetime(df[sort_by])
    except ValueError as e:
        raise ValueError(f"Failed to convert '{sort_by}' column to datetime.") from e

    labels = df[label_col].to_numpy()
    times = df[sort_by].to_numpy(dtype="datetime64[ns]")
    num_labels = len(labels)

    # Use DataFrame's TIME column if segmented_end_time_array is None
    if segmented_end_time_array is None:
        segmented_end_time_array = df[sort_by].to_numpy(dtype="datetime64[ns]")
    elif not np.issubdtype(segmented_end_time_array.dtype, np.datetime64):
        raise ValueError("segmented_end_time_array must contain datetime64 values.")

    # Vectorized computation
    shift_delta = np.timedelta64(shift_minutes, "m")
    target_times = segmented_end_time_array + shift_delta
    indices = np.searchsorted(times, target_times, side="left")
    valid_mask = indices < num_labels

    shifted_labels = np.full(len(target_times), np.nan)
    shifted_times = np.full(len(target_times), np.datetime64("NaT", 'ns'))
    shifted_labels[valid_mask] = labels[indices[valid_mask]]
    shifted_times[valid_mask] = times[indices[valid_mask]]

    if not valid_mask.any():
        raise ValueError("No valid windows with shifted labels.")

    return shifted_labels, shifted_times