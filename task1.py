import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def preprocess_data(df_train, df_test):
    # Prepare the training data
    X_train, y_train = prepare_data(df_train)

    # Prepare the test data
    X_test,_ = prepare_data(df_test, is_train=False)

    return X_train, y_train, X_test

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


def train_model(X_train, y_train):
    # Hyperparameter Tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='f1_macro', cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model

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
    # Make predictions on the test data using the best model
    y_test_pred = model.predict(X_test)

    # Create a DataFrame with the 'id' and 'cancellation' columns and save it
    output_df = pd.DataFrame({
        'id': X_test['h_booking_id'],
        'cancellation': y_test_pred
    })
    output_df.to_csv(output_filename, index=False)

def main():
    # Load the training dataset
    df_train = pd.read_csv('agoda_cancellation_train.csv')

    # Use a smaller subset of the data
    df_train = df_train.sample(n=1000, random_state=42)

    # Load the test dataset
    df_test = pd.read_csv('Agoda_Test_1.csv')

    # Use a smaller subset of the test data
    df_test = df_test.sample(n=500, random_state=42)

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

if __name__ == '__main__':
    main()
