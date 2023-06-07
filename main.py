import numpy as np
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

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


  # if isinstance(y, pd.Series):
  #       y = y[y > 0].dropna()
  #       X = X.loc[y.index]
  #   X = X.dropna().drop_duplicates()
  #   X = X.drop(["id", "date", "lat", "long", "sqft_lot15", "sqft_living15"], axis=1)
  #   for col in ["sqft_living", "sqft_lot"]:
  #       X = X[X[col] > 0]
  #   for col in ["bedrooms", "bathrooms", "floors", "sqft_basement", "sqft_above", "yr_built", "yr_renovated"]:
  #       X = X[X[col] >= 0]
  #   X["zipcode"] = X["zipcode"].astype(float)
  #   X = pd.get_dummies(X, columns=['zipcode'])
  #   X["renovated"] = np.where(X["yr_renovated"] > 0, 1, 0)
  #   X = X.drop("yr_renovated", axis=1)
  #   X = X.astype(float)
  #   if isinstance(y, pd.Series):
  #       y = y.astype(float)
  #       return X, y.loc[X.index]
  #   else:
  #       return X
