import numpy as np
import pandas as pd



def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,
                     parse_dates=
                     ["booking_datetime", "checkin_datetime", "checkout_datetime"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year"] = df["Year"].astype(str)
    df = df.dropna().drop_duplicates()
    df = df[df.Temp > -40]
    df = df[df.Temp < 60]
    return df
