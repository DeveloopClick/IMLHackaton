import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import plotly.express as px

def load_and_preprocess_data(filename: str):
    data = pd.read_csv(filename)
    data = data.dropna(axis=0)  # drop all rows with any NaN values
    relevant_data = data[['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_country_code',
                          'hotel_star_rating', 'charge_option', 'accommadation_type_name',
                          'customer_nationality', 'guest_nationality_country_name',
                          'guest_is_not_the_customer', 'no_of_room', 'origin_country_code',
                          'original_payment_currency', 'is_first_booking', 'request_airport',
                          'hotel_brand_code', 'hotel_chain_code', 'hotel_live_date', 'no_of_adults',
                          'no_of_children', 'is_user_logged_in']]
    y = data['original_selling_amount']
    for col in relevant_data.columns:
        # print pearson correlation coefficient
        print(col)
        print(relevant_data[col].corr(y))
    relevant_data = pd.get_dummies(relevant_data)  # one-hot encode categorical variables
    return relevant_data, y


def train_model(X, y):
    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=7)
    xgb_reg.fit(X, y)
    return xgb_reg


def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return rmse


# def predict_and_save_submission(model, data, filename):
#     data_preprocessed = preprocess_data(data)
#     predictions = model.predict(data_preprocessed)
#
#     submission = pd.DataFrame({'h_booking_id': data['h_booking_id'],
#                                'predicted_selling_amount': predictions})
#
#     submission.to_csv(filename, index=False)

if __name__ == '__main__':
    # Load and preprocess the training data
    print("Loading and preprocessing training data...")
    data, y = load_and_preprocess_data('agoda_cancellation_train.csv')

    # Split the training data
    print("Splitting training data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0., random_state=42)

    # Train the model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Calculate RMSE on the test data
    print("Calculating RMSE on the test set...")
    y_test_pred = model.predict(X_test)
    rmse_test = calculate_rmse(y_test, y_test_pred)
    print(f'Test RMSE: {rmse_test}')

    # # Load and preprocess the submission data
    # print("Loading and preprocessing submission data...")
    # submission_data = load_data('Agoda_Test_2.csv')
    #
    # # Predict and save submission
    # print("Predicting and saving submission...")
    # predict_and_save_submission(model, submission_data, 'agoda_cost_of_cancellation.csv')
    # print("Done!")
