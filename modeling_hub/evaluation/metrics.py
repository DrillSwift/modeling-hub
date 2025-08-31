
# Standard
from datetime import timedelta

# Third-Party
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Local
from modeling_hub.utils import get_abnormal_runs

def to_event_table(
    df_real: pd.DataFrame, df_pred: pd.DataFrame,
    time_col: str, 
    cat_col: str,
    normal_label: str,
    lookback_seconds: float = 900,  
    tolerance_seconds: float = 300, 
    min_overlap_seconds: float = 60
):
    """
    Evaluate if predicted abnormal runs correctly forecast real failures
    and also detect false alarms.

    Parameters
    ----------
    df_real : pd.DataFrame
        Ground truth dataframe with time & category.
    df_pred : pd.DataFrame
        Predicted dataframe with time & category.
    time_col : str
        Column containing timestamps.
    cat_col : str
        Column containing categories.
    normal_label : str
        Label considered normal (others treated as failures).
    lookback_seconds : float
        How many seconds before failure_start defines the lookback point.
    tolerance_seconds : float
        +/- tolerance (in seconds) around the lookback point.
    min_overlap_seconds : float
        Minimum overlap (in seconds) required between predicted run and window.

    Returns
    -------
    events_df : pd.DataFrame
        Event-based table of real failures with prediction match info.
        Columns: [category, failure_start, failure_end, failure_duration,
                  predicted, time_to_event, matched_pred_idx]

    false_alarms_df : pd.DataFrame
        Predicted abnormal runs that did not match any real failure.
        Columns: [category, pred_start, pred_end, pred_duration]
    """

    # Convert numeric arguments to timedeltas
    lookback = timedelta(seconds=lookback_seconds)
    tolerance = timedelta(seconds=tolerance_seconds)
    min_overlap = timedelta(seconds=min_overlap_seconds)

    # Get abnormal runs
    real_runs = get_abnormal_runs(df_real, time_col, cat_col, normal_label)
    pred_runs = get_abnormal_runs(df_pred, time_col, cat_col, normal_label)

    matched_pred_indices = set()
    results = []

    for _, row in real_runs.iterrows():
        failure_start, failure_end, real_cat = row["start"], row["end"], row["category"]

        # Lookback window centered at failure_start - lookback
        lb_point = failure_start - lookback
        window_start = lb_point - tolerance / 2
        window_end   = lb_point + tolerance / 2

        predicted = False
        time_to_event = None
        matched_idx = None

        for jdx, prow in pred_runs.iterrows():
            pred_start, pred_end, pred_cat = prow["start"], prow["end"], prow["category"]

            if pred_cat != real_cat:
                continue  # category must match

            # overlap with window
            overlap_start = max(pred_start, window_start)
            overlap_end   = min(pred_end, window_end)

            if overlap_end > overlap_start:
                if (overlap_end - overlap_start) >= min_overlap:
                    predicted = True
                    time_to_event = (failure_start - pred_start).total_seconds()
                    matched_idx = jdx
                    matched_pred_indices.add(jdx)
                    break

        results.append({
            "category": real_cat,
            "failure_start": failure_start,
            "failure_end": failure_end,
            "failure_duration": row["duration"].total_seconds(),
            "predicted": predicted,
            "time_to_event": time_to_event,
            "matched_pred_idx": matched_idx
        })

    events_df = pd.DataFrame(results)

    # False alarms = predicted runs not matched
    false_alarm_runs = pred_runs.loc[~pred_runs.index.isin(matched_pred_indices)].copy()
    false_alarm_runs.rename(columns={
        "start": "pred_start",
        "end": "pred_end",
        "duration": "pred_duration"
    }, inplace=True)

    false_alarms_df = false_alarm_runs[["category", "pred_start", "pred_end", "pred_duration"]]

    return events_df, false_alarms_df

def compute_event_metrics(events_df: pd.DataFrame, false_alarms_df: pd.DataFrame):
    """
    Compute event-level metrics from to_event_table outputs,
    both overall and per category.

    Parameters
    ----------
    events_df : pd.DataFrame
        Real events with prediction flags (from to_event_table).
        Must contain columns ["category", "predicted"].
    false_alarms_df : pd.DataFrame
        Predicted runs that did not match any real event (from to_event_table).
        Must contain ["category"].

    Returns
    -------
    dict with:
        - overall metrics
        - per-category metrics
    """

    TP = int(events_df["predicted"].sum())
    FN = int((~events_df["predicted"]).sum())
    FP = len(false_alarms_df)

    y_true = np.ones(len(events_df), dtype=int)
    y_pred = events_df["predicted"].astype(int).values

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])

    categories = sorted(events_df["category"].unique())
    category_metrics = {}

    for cat in categories:
        events_cat = events_df[events_df["category"] == cat]
        false_alarms_cat = false_alarms_df[false_alarms_df["category"] == cat]

        TP_c = int(events_cat["predicted"].sum())
        FN_c = int((~events_cat["predicted"]).sum())
        FP_c = len(false_alarms_cat)

        precision_c = TP_c / (TP_c + FP_c) if (TP_c + FP_c) > 0 else 0.0
        recall_c    = TP_c / (TP_c + FN_c) if (TP_c + FN_c) > 0 else 0.0
        f1_c        = (2 * precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0.0
        accuracy_c  = TP_c / (TP_c + FN_c) if (TP_c + FN_c) > 0 else 0.0

        category_metrics[cat] = {
            "TP": TP_c,
            "FP": FP_c,
            "FN": FN_c,
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "accuracy": accuracy_c,
        }

    return {
        "overall": {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist()
        },
        "by_category": category_metrics
    }

if __name__ == "__main__":

    df_real = pd.DataFrame({
        "time": pd.to_datetime([
            "2025-01-01 08:00:00",
            "2025-01-01 08:05:00",
            "2025-01-01 08:10:00",
            "2025-01-01 08:15:00",  # failure starts (Error)
            "2025-01-01 08:20:00",
            "2025-01-01 08:25:00",
            "2025-01-01 08:30:00",
            "2025-01-01 08:35:00",
            "2025-01-01 08:40:00",  # failure starts (Critical)
            "2025-01-01 08:45:00",
            "2025-01-01 08:50:00",
            "2025-01-01 08:55:00",
        ]),
        "category": [
            "Normal","Normal","Normal",
            "Error","Error","Error",   # Error run
            "Normal","Normal",
            "Critical","Critical",     # Critical run
            "Normal","Normal"
        ]
    })

    df_pred = pd.DataFrame({
        "time": pd.to_datetime([
            "2025-01-01 07:30:00",
            "2025-01-01 07:35:00",
            "2025-01-01 07:40:00",
            "2025-01-01 07:45:00",
            "2025-01-01 07:50:00",
            "2025-01-01 07:55:00",
            "2025-01-01 08:00:00",
            "2025-01-01 08:25:00",
            "2025-01-01 08:30:00",
            "2025-01-01 09:15:00",
            "2025-01-01 09:20:00",
            "2025-01-01 09:25:00",
        ]),
        "category": [
            "Normal",
            "Error","Error",  
            "Normal","Normal","Normal",
            "Normal",
            "Critical","Critical",
            "Normal","Normal","Normal"
        ]
    })

    print(df_real)
    print(df_pred)

    results = to_event_table(
        df_real, df_pred,
        "time", "category", "Normal",
        lookback_seconds=900,     
        tolerance_seconds=300,     
        min_overlap_seconds=60    
    )
    print(json.dumps(compute_event_metrics(*results), indent=4))

