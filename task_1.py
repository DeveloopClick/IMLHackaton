import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_data(df_train, df_test):
    # Prepare the training data
    X_train, y_train = prepare_data(df_train)

    # Prepare the test data
    X_test = prepare_data(df_test, is_train=False)

    return X_train, y_train, X_test


def prepare_data(df, is_train=True):
    # Specify the relevant columns to keep

    relevant_cols = ['hotel_country_code',
                     'hotel_star_rating', 'charge_option', 'accommadation_type_name',
                     'customer_nationality', 'guest_nationality_country_name',
                     'guest_is_not_the_customer', 'no_of_room', 'origin_country_code',
                     'original_payment_currency', 'is_first_booking', 'request_airport',
                     'hotel_brand_code',
                     'no_of_children', 'is_user_logged_in']
    if is_train:
        relevant_cols.append('cancellation_datetime')

    df = df[relevant_cols].drop_duplicates()
    # Prepare the data
    # X = df.copy()
    if is_train:
        df["y"] = df['cancellation_datetime'].notnull().astype(int)
        df = df.drop('cancellation_datetime', axis=1)
        df = df.dropna()
        y = df["y"]
        X = df.drop(columns=["y"])
    else:
        X = df.dropna()
        y=None




    # Perform one-hot encoding on categorical columns
    cat_cols = ['hotel_country_code', 'charge_option', 'accommadation_type_name',
                'customer_nationality', 'guest_nationality_country_name',
                'origin_country_code', 'original_payment_currency',
                'hotel_brand_code']

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

def train_model(X_train, y_train):
    # Train a Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_val, y_val):
    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    print("Validation Accuracy:", accuracy)
    print("Validation Precision:", precision)
    print("Validation Recall:", recall)
    print("Validation F1 Score:", f1)

def predict_test_data(model, X_test, output_filename):
    # Make predictions on the test data using the trained model
    y_test_pred = model.predict(X_test)

    # Create a DataFrame with the 'id' and 'cancellation' columns and save it
    output_df = pd.DataFrame({
        'id': X_test['h_booking_id'],
        'cancellation': y_test_pred
    })
    output_df.to_csv(output_filename, index=False)

def run_task_1():
    # Load the training dataset
    df_train = pd.read_csv('agoda_cancellation_train.csv')

    # Use a smaller subset of the data
    df_train = df_train.sample(n=4000, random_state=42)

    # Load the test dataset
    df_test = pd.read_csv('Agoda_Test_1.csv')

    # Use a smaller subset of the test data
    df_test = df_test.sample(n=1000, random_state=42)

    # Preprocess the data
    X_train, y_train, X_test = preprocess_data(df_train, df_test)

    # Train the model
    model = train_model(X_train, y_train)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Predict on the test data and save the output
    predict_test_data(model, X_test, 'agoda_cancellation_prediction.csv')

