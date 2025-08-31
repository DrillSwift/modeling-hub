
# Standard

# Third-Party
import pandas as pd

# Local

def get_abnormal_runs(df: pd.DataFrame, time_col: str, cat_col: str, normal_label: str, split_by_category: bool=True):
    """
    Detect consecutive runs of abnormal categories in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least time and category columns.
    time_col : str
        Name of the timestamp column.
    cat_col : str
        Name of the category column.
    normal_label : str
        Label that indicates normal rows. Everything else is abnormal.
    split_by_category : bool, default=True
        - If True: treat different abnormal categories separately.
        - If False: merge all abnormal categories into a single "Not-Normal" run.
    
    Returns
    -------
    pd.DataFrame with columns [start, end, category]
    """
    df = df.copy()

    if split_by_category:
        df["group"] = (df[cat_col] != df[cat_col].shift()).cumsum()
        df["run_category"] = df[cat_col]
    else:
        df["is_abnormal"] = df[cat_col] != normal_label
        df["group"] = (df["is_abnormal"] != df["is_abnormal"].shift()).cumsum()
        df["run_category"] = df["is_abnormal"].map(lambda x: "Not-Normal" if x else normal_label)

    runs = (
        df.groupby("group")
          .agg(start=(time_col, "first"),
               end=(time_col, "last"),
               category=("run_category","first"))
          .reset_index(drop=True)
    )

    runs = runs[runs["category"] != normal_label].reset_index(drop=True)
    runs["duration"] = runs["end"] - runs["start"]

    return runs[["category", "start", "end", "duration"] ]
