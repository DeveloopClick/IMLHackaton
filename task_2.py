import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from forex_python.converter import CurrencyRates

CONVERSION_RATES_TO_USD = {"CNY": 0.141145, "ZAR": 0.05125604, "KRW": 0.00076561928, "SGD": 0.74047095,
                             "THB": 0.028749006, "ARS": 0.0041429069, "TWD": 0.032567006, "SAR": 0.26666667,
                             "USD": 1.00, "MYR": 0.21845246, "SEK": 0.092128658, "NZD": 0.60573548,
                           "VND": 0.000042595541, "OMR": 2.599953, "UAH": 0.027277709, "KZT": 0.00222717,
                           "XPF": 0.0089806655, "KHR": 0.000243195, "LAK": 0.000055590889}


def load_and_preprocess_data(filename: str):
    data = pd.read_csv(filename)
    # change currency and price to match the usd price
    convert_price_to_usd(data)
    relevant_data = data[[
                          'booking_datetime', 'checkin_date', 'checkout_date', 'hotel_star_rating', 'charge_option', 'accommadation_type_name',
                          'no_of_room'
                          ]]
    # one hot encode for country_code, acoommadation_type_name, charge_option
    relevant_data = pd.get_dummies(relevant_data, columns=['charge_option'])
    relevant_data = pd.get_dummies(relevant_data, columns=['accommadation_type_name'])

    # convert date/time features to numerical
    for feature in ['checkin_date', 'checkout_date']:
        relevant_data[feature] = pd.to_datetime(relevant_data[feature])
        relevant_data[feature + '_month'] = relevant_data[feature].dt.month
        relevant_data[feature + '_day'] = relevant_data[feature].dt.day
        relevant_data[feature + '_weekday'] = relevant_data[feature].dt.weekday

    # calculate number of nights
    relevant_data['num_nights'] = (relevant_data['checkout_date'] - relevant_data['checkin_date']).dt.days
    relevant_data.drop(['checkin_date', 'checkout_date'], axis=1, inplace=True)

    if 'booking_datetime' in relevant_data.columns:
        relevant_data.drop('booking_datetime', axis=1, inplace=True)

    relevant_data = relevant_data.dropna(axis=0)
    y = data['original_selling_amount_usd']
    return relevant_data, y


def train_model(X, y):
    xgb_reg = xgb.XGBRegressor()
    parameters = {'n_estimators': [50, 100],
                  'learning_rate': [0.05, 0.1],
                  'max_depth': [5, 7],
                  'gamma': [0, 0.1],
                  'subsample': [0.75, 1],
                  'colsample_bytree': [0.75, 1]}

    xgb_grid = GridSearchCV(xgb_reg,
                            parameters,
                            cv = 3,
                            n_jobs = 5,
                            verbose=True)

    xgb_grid.fit(X, y)
    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)

    return xgb_grid.best_estimator_


def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return rmse


def get_conversion_rates(currencies):
    cr = CurrencyRates()
    rates = {}
    for currency in currencies:
        try:
            rates[currency] = cr.get_rate(currency, 'USD')
        except:
            rates[currency] = CONVERSION_RATES_TO_USD.get(currency)
    return rates


def convert_price_to_usd(data_in_all_currencys):
    # change currency to usd
    rates = get_conversion_rates(data_in_all_currencys['original_payment_currency'].unique())
    data_in_all_currencys['original_selling_amount_usd'] = data_in_all_currencys.apply(
        lambda row: row['original_selling_amount'] * rates[row['original_payment_currency']], axis=1)

def load_and_preprocess_test_data(filename: str):
    data = pd.read_csv(filename)
    relevant_data = data[[
                          'booking_datetime', 'checkin_date', 'checkout_date', 'hotel_star_rating', 'charge_option', 'accommadation_type_name',
                          'no_of_room'
                          ]]
    # one hot encode for country_code, acoommadation_type_name, charge_option
    relevant_data = pd.get_dummies(relevant_data, columns=['charge_option'])
    relevant_data = pd.get_dummies(relevant_data, columns=['accommadation_type_name'])

    # convert date/time features to datetime
    relevant_data['checkin_date'] = pd.to_datetime(relevant_data['checkin_date'])
    relevant_data['checkout_date'] = pd.to_datetime(relevant_data['checkout_date'])

    # calculate number of nights
    relevant_data['num_nights'] = (relevant_data['checkout_date'] - relevant_data['checkin_date']).dt.days
    relevant_data.drop(['checkin_date', 'checkout_date', 'booking_datetime'], axis=1, inplace=True)

    relevant_data = relevant_data.dropna(axis=0)
    return relevant_data

def predict_and_save(model, filename):
    # Load and preprocess the test data similar to how we preprocessed the training data
    data_test = load_and_preprocess_test_data(filename)

    # Predict the selling amounts in USD
    predicted_selling_amount_usd = model.predict(data_test)

    # Note: No conversion of currencies here as we don't know the original_payment_currency

    # Load the test data again to get the h_booking_id for each record
    data_test_original = pd.read_csv(filename)

    # Create a DataFrame with h_booking_id and predicted_selling_amount
    output = pd.DataFrame({
        'h_booking_id': data_test_original['h_booking_id'],
        'predicted_selling_amount': predicted_selling_amount_usd
    })

    # Note: If you have a method to determine whether an order is likely to be cancelled,
    # you can implement it here. For instance, if order_is_likely_to_be_cancelled() returns
    # True or False, you could use:
    # output.loc[order_is_likely_to_be_cancelled(data_test_original), 'predicted_selling_amount'] = -1

    # Save the DataFrame to a CSV file
    output.to_csv('agoda_cost_of_cancellation.csv', index=False)

    return output




if __name__ == '__main__':
    # Load and preprocess the training data
    print("Loading and preprocessing training data...")
    data, y = load_and_preprocess_data('agoda_cancellation_train.csv')

    # Split the training data
    print("Splitting training data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Calculate RMSE on the test data
    print("Calculating RMSE on the test set...")
    y_test_pred = model.predict(X_test)

    rmse_test = calculate_rmse(y_test, y_test_pred)
    print(f'Test RMSE: {rmse_test}')

    # Predict and save the predicted selling amounts for the test data
    print("Predicting and saving the selling amounts for the test data...")
    predict_and_save(model, 'Agoda_Test_2.csv')
