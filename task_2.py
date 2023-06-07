import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from math import sqrt


def load_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename)
    return data


def preprocess_data(data):
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    data = preprocessor.fit_transform(data)

    return data, preprocessor


def train_model(X_train, y_train):
    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def calculate_rmse(y_test, y_pred):
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    return rmse


def predict_and_save_submission(clf, preprocessor, data, filename):
    data_preprocessed = preprocessor.transform(data)
    data['predicted_selling_amount'] = clf.predict(data_preprocessed)
    submission = data[['h_booking_id', 'predicted_selling_amount']]
    submission.to_csv(filename, index=False)

if __name__ == '__main__':

    # Load data
    data = load_data('agoda_cancellation_train.csv')

    # Separate features and target variable
    features = data.drop(['h_booking_id', 'cancellation_datetime', 'original_selling_amount'], axis=1)
    y = data['original_selling_amount']

    # Preprocess the features
    preprocessed_features, preprocessor = preprocess_data(features)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, y, test_size=0.2, random_state=0)

    # Train the model
    clf = train_model(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Calculate RMSE
    rmse = calculate_rmse(y_test, y_pred)
    print(f'RMSE: {rmse}')

    # Predict on whole dataset for submission
    predict_and_save_submission(clf, preprocessor, features, 'agoda_cost_of_cancellation.csv')
