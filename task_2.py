import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tqdm import tqdm


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


def train_model(X, y):
    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf


def predict_and_save_submission(clf, preprocessor, data, filename):
    data_preprocessed = preprocessor.transform(data)
    predictions = clf.predict(data_preprocessed)

    # Create a dataframe for submission
    submission = pd.DataFrame({'h_booking_id': data['h_booking_id'],
                               'predicted_selling_amount': predictions})

    submission.to_csv(filename, index=False)

if __name__ == '__main__':

    # Load and preprocess the training data
    print("Loading and preprocessing training data...")
    train_data = load_data('agoda_cancellation_train.csv')
    features = train_data.drop(['h_booking_id', 'cancellation_datetime', 'original_selling_amount'], axis=1)
    y = train_data['original_selling_amount']
    preprocessed_features, preprocessor = preprocess_data(features)

    # Train the model
    print("Training model...")
    clf = train_model(preprocessed_features, y)

    # Load and preprocess the test data
    print("Loading and preprocessing test data...")
    test_data = load_data('Agoda_Test_2.csv')

    # Predict and save submission
    print("Predicting and saving submission...")
    predict_and_save_submission(clf, preprocessor, test_data, 'agoda_cost_of_cancellation.csv')
    print("Done!")
