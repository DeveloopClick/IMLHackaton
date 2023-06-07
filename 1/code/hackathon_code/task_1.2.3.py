# from task_1 import load_data # TODO: change this to its final location
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
LINES_TO_RUN = 10


def agoda_churn_prediction_model(filename: str) -> None:
    """
    Build a churn prediction model for Agoda.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    """
    date_columns = ["booking_datetime", "checkin_date", "checkout_date", 
                    "hotel_live_date", "cancellation_datetime"]
    useless_columns = ["h_booking_id", "hotel_id"]
    useful_columns = []
    df = pd.read_csv(filename, parse_dates=date_columns)
    current_date = pd.Timestamp.now().normalize()
    date_columns.pop()
    # create numerical values for dates, for every date column
    for date_column in date_columns:
        df[date_column] = (current_date - df[date_column]).dt.days

    # drop columns that are not useful for prediction
    df = df.drop(columns=useless_columns)
    df = df.drop(columns=date_columns)

    # Take the first LINE_TO_RUN lines
    df = df.head(LINES_TO_RUN)
    # The y to train our model on will be 1 if the booking was cancelled, 0 otherwise
    y = df["cancellation_datetime"].notnull().astype(int)

    # The X to train our model on will be the following features
    X = df.drop(columns=["cancellation_datetime"])
    # Feature name list
    feature_names = X.columns

    # Get the scores and feature names
    selector = SelectKBest(score_func=chi2, k=2)
    X_new = selector.fit_transform(X, y)

    # Get the scores and feature names
    scores = selector.scores_
    feature_names = X.columns

    # Create a DataFrame to store the scores
    scores_df = pd.DataFrame({'Feature': feature_names, 'Score': scores})
    scores_df = scores_df.sort_values(by='Score', ascending=False)

    # Generate a bar plot to visualize the feature importance scores
    plt.figure(figsize=(6, 6))
    plt.bar(range(len(scores)), scores, tick_label=feature_names)
    plt.xticks(rotation=43)
    plt.xlabel('Features')
    plt.ylabel('Score')
    plt.title('Feature Importance Scores')
    plt.show()

    # Print the selected features
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print('Selected Features:', selected_features)
def prepare_data(df, is_train=True):
    # Fill NaN values with defaults
    default_values = {
        'hotel_country_code': 'other',
        'accommadation_type_name': 'other',
        'charge_option': 'other',
        'customer_nationality': 'other',
        'guest_nationality_country_name': 'other',
        'origin_country_code': 'other',
        'language': 'other',
        'original_payment_method': 'other',
        'original_payment_type': 'other',
        'original_payment_currency': 'other',
        'cancellation_policy_code': 'other',
        'hotel_area_code': 'other',
        'hotel_brand_code': 'other',
        'hotel_chain_code': 'other',
        'hotel_city_code': 'other'
    }
    df = df.fillna(default_values)

    # Prepare the data
    if is_train:
        df = df.dropna(subset=['cancellation_datetime'])
    X = df.drop(['cancellation_datetime'], axis=1)
    y = df['cancellation_datetime'].notnull().astype(int)

    # Convert date and time columns to numerical representation
    date_cols = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date']
    for col in date_cols:
        X[col] = (pd.to_datetime(X[col]).astype('int64') // 10**9).astype('int32')

    # Convert categorical columns to strings
    cat_cols = ['hotel_country_code', 'accommadation_type_name', 'charge_option',
                'customer_nationality', 'guest_nationality_country_name',
                'origin_country_code', 'language', 'original_payment_method',
                'original_payment_type', 'original_payment_currency',
                'cancellation_policy_code', 'hotel_area_code', 'hotel_brand_code',
                'hotel_chain_code', 'hotel_city_code']

    for col in cat_cols:
        X[col] = X[col].astype(str)

    # Apply label encoders
    label_encoders = {}
    for col in cat_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    if is_train:
        # Handle unseen labels in the categorical columns
        for col in cat_cols:
            X[col] = np.where(X[col].isin(label_encoders[col].classes_), X[col], 'other')

        # Handle unseen labels in the target variable
        valid_labels = set(label_encoders['cancellation_policy_code'].classes_)
        valid_indices = y.map(lambda x: x in valid_labels)
        X = X[valid_indices]
        y = y[valid_indices]

    return X, y

# TODO: delete this
if __name__ == '__main__':
    agoda_churn_prediction_model(
        "/Users/staveyal/code/university/IMLHackaton/agoda_cancellation_train.csv"
        )