import pandas as pd
import numpy as np

def load_openaq(path: str = "data/openaq_2025.csv") -> pd.DataFrame:
    """Read raw OpenAQ export and preprocess into wide table with six variables.

    This replicates the transformation used by the dashboard so that
    standalone scripts can operate on the same cleaned dataset.
    """
    raw = pd.read_csv(path)
    param_map = {
        "pm25": "pm25",
        "pm10": "pm10",
        "no2": "no2",
        "o3": "ozone",
        "temperature": "temperature",
        "relativehumidity": "humidity",
    }
    raw = raw[raw["parameter"].isin(param_map.keys())].copy()
    raw["variable"] = raw["parameter"].map(param_map)
    # Parse datetimes with UTC to handle mixed timezone offsets cleanly
    raw["datetime"] = pd.to_datetime(raw["datetimeLocal"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["datetime"])
    wide = (
        raw.pivot_table(
            index=["location_id", "location_name", "datetime"],
            columns="variable",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None
    unique_locs = wide[["location_id", "location_name"]].drop_duplicates().sort_values("location_id")
    unique_locs["station"] = [f"Station_{i+1}" for i in range(len(unique_locs))]
    df = wide.merge(unique_locs, on=["location_id", "location_name"], how="left")
    cols = ["station", "datetime", "pm25", "pm10", "no2", "ozone", "temperature", "humidity"]
    df = df[cols]
    df = df.dropna(how="all", subset=["pm25", "pm10", "no2", "ozone", "temperature", "humidity"])
    df["zone"] = df["station"].apply(lambda x: "Industrial" if int(x.split("_")[1]) <= 50 else "Residential")
    
    # Convert timezone-aware timestamps to tz-naive UTC (remove timezone info)
    if df["datetime"].dtype != 'datetime64[ns]':  # if it's still tzaware
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    
    df = df.dropna(subset=["datetime"])  # drop rows where datetime is invalid
    
    # extract components
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    return df
