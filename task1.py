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
    df = pd.read_csv(filename, parse_dates=["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year"] = df["Year"].astype(str)
    df = df.dropna().drop_duplicates()
    df = df[df.Temp > -40]
    df = df[df.Temp < 60]
    return df


if __name__ == '__main__':
