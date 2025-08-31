
# Standard

# Third-Party
import pandas as pd

# Local
from stabilizer import EventStabilizer

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