
# Standard
from abc import ABC, abstractmethod

# Third-Party
import pandas as pd
import numpy as np

# Local
from modeling_hub.utils import get_abnormal_runs

class Stabilizer(ABC):
    """
    Abstract base class for stabilizing time-series classification outputs.

    A stabilizer transforms noisy, high-frequency predictions (e.g., per-second
    sensor labels) into robust, operator-friendly signals. 

    Subclasses must implement the `stabilize` method to apply a concrete
    stabilization policy.

    Methods
    -------
    stabilize(data : pd.DataFrame) -> pd.DataFrame
        Apply the stabilization logic to the input DataFrame and return a
        modified DataFrame with one or more new columns (e.g., "<label>_stabilized").
    """
    @abstractmethod
    def stabilize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stabilization to raw prediction data.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing at least a timestamp column and one or more
            columns with raw predicted class labels.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with one or more additional columns containing
            stabilized predictions, according to the concrete implementation's

        Notes
        -----
        - This method must be implemented by all subclasses.
        - Typical usage is to reduce flickering in high-frequency predictions and
          produce operator-friendly event-level signals.
        """
        raise NotImplementedError("This method must be implemented by subclass.")

class EventStabilizer(Stabilizer):
    """
    Stabilizer that converts raw, high-frequency predictions into stable
    event-level signals.

    This implementation applies common post-processing rules to reduce flicker
    in streaming classification outputs:

      • Gap-fill: closes short NORMAL gaps between two identical failure runs.
      • Min-on: discards episodes shorter than a minimum duration.
      • Cooldown: extends the end of valid episodes to avoid rapid oscillations.
      • Priority resolution: when multiple classes overlap, the highest severity wins.

    Parameters
    ----------
    time_col : str
        Name of the timestamp column in the input DataFrame.
    label_col : str
        Name of the column containing raw class predictions.
    classes_order : list of str
        Classes ordered from highest to lowest severity (e.g. ["CRITICAL","MAJOR","MINOR"]).
    normal_label : str, default="NORMAL"
        Label representing the absence of failure.
    gap_fill_s : float, default=5.0
        Maximum duration of a gap (in seconds) to fill between two runs.
    min_on_s : float, default=60.0
        Minimum duration (in seconds) required for an episode to be kept.
    cooldown_s : float, default=15.0
        Extra duration (in seconds) to extend the end of valid episodes.

    Methods
    -------
    stabilize(data : pd.DataFrame) -> pd.DataFrame
        Applies stabilization to the input DataFrame and returns it with an
        additional column `<label_col>_stabilized` containing the stable
        predictions.
    """
    def __init__(self, 
                 time_col: str, 
                 target_col: str, 
                 classes_order: list[str], 
                 normal_label: str,
                 gap_fill_s: float=5.0,
                 min_on_s: float=60.0,
                #  cooldown_s: float=15.0
                 ):
        super().__init__()

        self._time_col = time_col
        self._target_col = target_col
        self._classes_order = classes_order
        self._normal_label = normal_label
        self._gap_fill_s = gap_fill_s
        self._min_on_s = min_on_s
        # self._cooldown_s = cooldown_s

    def _ensure_time_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Convert the time column to a pandas.DatetimeIndex and validate ordering.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing the time column specified by `self._time_col`.

        Returns
        -------
        pd.Series
            A pandas Series of datetime values corresponding to the time column.

        Raises
        ------
        ValueError
            If the time column is not strictly increasing (i.e., unsorted or contains
            duplicates that break monotonicity).

        Notes
        -----
        - Ensures that all stabilization logic operates on clean, ordered time data.
        - Subclasses rely on this check before applying gap-fill, min-on, or cooldown.
        """
        t = pd.to_datetime(data[self._time_col], utc=False)
        if not t.is_monotonic_increasing:
            raise ValueError(f"{self._time_col} must be sorted ascending.")
        return t

    def _runs(self, mask: pd.Series, time: pd.Series) -> pd.DataFrame:
        """
        Identify consecutive runs of identical boolean values in a mask.

        Parameters
        ----------
        mask : pd.Series
            Boolean Series indicating whether the target class is active at each row.
        time : pd.Series
            Series of datetime values aligned with `mask`.

        Returns
        -------
        pd.DataFrame
            Table of contiguous runs with the following columns:
            - start_idx : int
                Index of the first row in the run.
            - end_idx : int
                Index of the last row in the run.
            - value : bool
                The run value (True or False).
            - start_time : pd.Timestamp
                Timestamp of the first row in the run.
            - end_time : pd.Timestamp
                Timestamp of the last row in the run.
            - duration_s : float
                Run duration in seconds (end_time - start_time).

        Notes
        -----
        - A "run" is defined as consecutive rows where `mask` does not change.
        - Useful for gap-filling, minimum-duration enforcement, and cooldown extension,
        since all of these operate on contiguous runs rather than single rows.
        """

        rid = (mask != mask.shift(1, fill_value=mask.iloc[0])).cumsum()

        grp = mask.groupby(rid)
        start_idx = grp.apply(lambda s: s.index[0])
        end_idx   = grp.apply(lambda s: s.index[-1])
        value     = grp.first()

        start_time = start_idx.map(time.__getitem__)
        end_time   = end_idx.map(time.__getitem__)
        duration_s = (end_time - start_time).dt.total_seconds()

        runs = pd.DataFrame({
            "start_idx": start_idx.values,
            "end_idx": end_idx.values,
            "value": value.values,
            "start_time": start_time.values,
            "end_time": end_time.values,
            "duration_s": duration_s.values
        })
        return runs.reset_index(drop=True)

    def _gap_fill(self, mask: pd.Series, time: pd.Series) -> pd.Series:
        """
        Apply gap-filling to smooth short interruptions in an active run.

        Parameters
        ----------
        mask : pd.Series
            Boolean Series indicating whether the target class is active at each row.
        time : pd.Series
            Series of datetime values aligned with `mask`.

        Returns
        -------
        pd.Series
            A copy of the input mask with short False runs converted to True.

        Notes
        -----
        - A False run is eligible for filling if:
            1. It is sandwiched between two True runs, and
            2. The time gap between the end of the preceding True run and the start
            of the following True run is less than or equal to `self._gap_fill_s`
            seconds.
        - This prevents brief NORMAL blips (sensor noise, transient errors) from
        splitting what should be treated as one continuous episode.
        - Example: if `gap_fill_s = 5`, and the model predicts
        True → False → True with only a 3-second gap, the False section will be
        filled as True, merging both True runs into a single episode.
        """
        runs = self._runs(mask, time)
        filled = mask.copy()

        for i in range(1, len(runs) - 1):
            r = runs.iloc[i]
            if not r["value"]:
                prev_true = runs.iloc[i - 1]
                next_true = runs.iloc[i + 1]
                if prev_true["value"] and next_true["value"]:
                    gap = (next_true["start_time"] - prev_true["end_time"]).total_seconds()
                    if gap <= self._gap_fill_s:
                        filled.iloc[r["start_idx"]: r["end_idx"] + 1] = True
        return filled

    def _apply_min_on(self, mask: pd.Series, time: pd.Series) -> pd.Series:
        """
        Enforce a minimum-on rule to remove short-lived runs.

        Parameters
        ----------
        mask : pd.Series
            Boolean Series indicating whether the target class is active at each row.
        time : pd.Series
            Series of datetime values aligned with `mask`.

        Returns
        -------
        pd.Series
            A copy of the input mask where True runs shorter than `self._min_on_s`
            seconds have been dropped (set to False).

        Notes
        -----
        - This ensures that only episodes of sufficient duration are considered valid.
        - Any run where the duration is strictly less than `self._min_on_s` is
        removed, helping to filter out noise and transient spikes.
        - Example: with `min_on_s = 30`, a run lasting 10 seconds will be discarded,
        while a run lasting 35 seconds will be retained.
        - Runs with duration exactly equal to `min_on_s` are preserved.
        """
        runs = self._runs(mask, time)
        keep = mask.copy()
        if runs.empty:
            return keep & False

        for _, r in runs.iterrows():
            if r["value"] and r["duration_s"] < self._min_on_s:
                keep.iloc[r["start_idx"]: r["end_idx"] + 1] = False
        return keep

    # def _extend_and_merge(self, mask: pd.Series, time: pd.Series) -> pd.Series:
    #     """
    #     Apply a cooldown extension to qualifying runs and merge overlaps.

    #     Parameters
    #     ----------
    #     mask : pd.Series
    #         Boolean Series indicating whether the target class is active at each row.
    #     time : pd.Series
    #         Series of datetime values aligned with `mask`.

    #     Returns
    #     -------
    #     pd.Series
    #         A boolean Series where:
    #         - Each True run lasting at least `self._min_on_s` seconds is extended by
    #         `self._cooldown_s` seconds beyond its end.
    #         - Overlapping or touching runs after extension are merged into a single
    #         continuous run.

    #     Notes
    #     -----
    #     - The cooldown helps prevent rapid oscillations by keeping the signal active
    #     for a buffer period after a valid run ends.
    #     - Only runs that meet the minimum duration threshold (`min_on_s`) are
    #     extended. Short runs are ignored.
    #     - Example: with `min_on_s=60` and `cooldown_s=15`, a 90-second run ending at
    #     01:00:00 will be extended to 01:00:15. If another qualifying run starts at
    #     01:00:10, the two intervals overlap and will be merged into a single run.
    #     """
    #     runs = self._runs(mask, time)
    #     intervals = []
    #     for _, r in runs.iterrows():
    #         if r["value"] and (r["duration_s"] >= self._min_on_s):
    #             s = r["start_time"]
    #             e = r["end_time"] + pd.Timedelta(seconds=self._cooldown_s)
    #             intervals.append((s, e))

    #     if not intervals:
    #         return pd.Series(False, index=mask.index)

    #     intervals.sort(key=lambda x: x[0])
    #     merged = [intervals[0]]
    #     for s, e in intervals[1:]:
    #         ls, le = merged[-1]
    #         if s <= le:
    #             merged[-1] = (ls, max(le, e))
    #         else:
    #             merged.append((s, e))

    #     out = pd.Series(False, index=mask.index)
    #     t = time
    #     for s, e in merged:
    #         out[(t >= s) & (t <= e)] = True
    #     return out

    def _stabilize_class(self, data: pd.DataFrame, target_class: str) -> pd.Series:
        """
        Construct a stabilized boolean mask for a single target class.

        This method applies the full stabilization pipeline to raw, row-level
        predictions of `target_class`:

        1. Gap-fill: convert short False runs into True if they are sandwiched
            between True runs and the inter-run gap is ≤ `self._gap_fill_s` seconds.
        2. Min-on: discard True runs shorter than `self._min_on_s` seconds.
        3. Cooldown: extend the end of remaining runs by `self._cooldown_s`
            seconds and merge overlapping runs.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing at least the time column (`self._time_col`)
            and the target label column (`self._target_col`).
        target_class : str
            Class label for which the stabilized mask is to be computed.

        Returns
        -------
        pd.Series
            Boolean Series indexed like `data`, where True indicates rows belonging
            to a stabilized run of `target_class`.

        Notes
        -----
        - Assumes the DataFrame is sorted by the time column.
        - Intended as a building block for multiclass stabilization, where the masks
        of multiple classes are combined with severity priority rules.
        """
        t = self._ensure_time_series(data)
        init_mask = (data[self._target_col] == target_class)

        m1 = self._gap_fill(init_mask, t)
        m2 = self._apply_min_on(m1, t)
        # m3 = self._extend_and_merge(m2, t)

        return m2

    def stabilize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Stabilize multiclass predictions into operator-friendly event signals.

        This method applies the stabilization pipeline to all classes in
        `self._classes_order` (from highest to lowest severity) and resolves
        overlaps by priority — the first (highest severity) class takes precedence
        when multiple masks are active at the same time.

        The result is a new column appended to the DataFrame:
        `<target_col>_stabilized`, which contains the final stabilized label at
        each row.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing at least:
            - A time column (`self._time_col`)
            - A column with raw predicted labels (`self._target_col`)

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with an additional column named
            `<target_col>_stabilized` containing the stabilized predictions.

        Notes
        -----
        - Stabilization rules include gap-filling, minimum duration (min_on),
        cooldown extension, and severity-priority resolution.
        - The DataFrame is copied before the new column is added to avoid
        mutating the caller’s data in place.
        - Example: if `self._target_col = "pred"`, the stabilized column will
        be `"pred_stabilized"`.
        """
        self._ensure_time_series(data)
        out = pd.Series(self._normal_label, index=data.index, dtype=object)

        for target_class in self._classes_order:  # highest → lowest severity
            m = self._stabilize_class(data, target_class)
            out = np.where(m, target_class, out)

        stabilized = pd.Series(out, index=data.index, name=f"{self._target_col}_stabilized")
        data = data.copy()
        data[stabilized.name] = stabilized
        return data

if __name__ == "__main__":
    
    df = pd.DataFrame({
    "time": pd.to_datetime([

        "2025-01-01 00:00:00",  # NORMAL

        "2025-01-01 00:00:05",  # MAJOR start
        "2025-01-01 00:00:10",  # MAJOR
        "2025-01-01 00:00:15",  # MAJOR
        "2025-01-01 00:00:20",  # MAJOR
        "2025-01-01 00:00:25",  # MAJOR
        "2025-01-01 00:00:30",  # NORMAL (short gap, 4s to next MAJOR → will be gap-filled with 5s)
        "2025-01-01 00:00:35",  # MAJOR (interleaving lower severity during MAJOR)
        "2025-01-01 00:00:40",  # MAJOR
        "2025-01-01 00:00:45",  # MAJOR
        "2025-01-01 00:00:50",  # MAJOR
        "2025-01-01 00:00:55",  # MAJOR
        "2025-01-01 00:01:00",  # MAJOR
        "2025-01-01 00:01:05",  # MAJOR  (from 00:00:05 → 00:01:05 = 60s exact)
        "2025-01-01 00:01:10",  # MAJOR  (now >60s, safely passes min_on=60)

        "2025-01-01 00:01:26",  # MINOR start (outside MAJOR’s cooldown by 1s if cooldown_s=15)
        "2025-01-01 00:01:35",  # MINOR
        "2025-01-01 00:01:45",  # MINOR
        "2025-01-01 00:01:55",  # MINOR
        "2025-01-01 00:02:05",  # MINOR
        "2025-01-01 00:02:15",  # MINOR
        "2025-01-01 00:02:30",  # MINOR (from 00:01:26 → 00:02:30 ~ 64s)
        "2025-01-01 00:02:45",  # NORMAL (end of MINOR; then cooldown extends a bit further)
    ]),
    "pred": [
        "NORMAL",

        "MAJOR","MAJOR", "MAJOR","MAJOR","MAJOR","NORMAL","MAJOR","MAJOR","MAJOR","MAJOR","MAJOR","MAJOR","MAJOR","MAJOR",

        "MINOR","MINOR","MINOR","MINOR","MINOR","MINOR","MINOR","NORMAL"
    ]
    })


    print(df)

    stabilizer = EventStabilizer(
        time_col="time",
        target_col="pred",
        classes_order=["NORMAL","MAJOR","MINOR"],
        normal_label="NORMAL",
        gap_fill_s=10, min_on_s=60,
        # cooldown_s=15,
    )

    df = stabilizer.stabilize(df)

    print(df)

    print("\n\n\n#################\n\n\n")    

    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01 08:00", periods=14, freq="h"),
        "category": [
            "Normal","Normal","Error","Error","Warning",
            "Normal","Warning","Warning","Warning","Normal",
            "Critical","Critical","Normal","Error"
        ]
    })

    print("Original Data:")
    print(df)

    # Separate abnormal categories
    print("\nSeparate runs by category:")
    print(get_runs(df, time_col="time", cat_col="category", normal_label="Normal", split_by_category=True))

    # Merge all abnormal categories into 'Not-Normal'
    print("\nAll abnormal merged as 'Not-Normal':")
    print(get_runs(df, time_col="time", cat_col="category", normal_label="Normal", split_by_category=False))