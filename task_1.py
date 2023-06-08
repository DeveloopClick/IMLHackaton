import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline

def preprocess_data(df_train, df_test, date_cols, cat_cols, num_cols,relevant_cols):

    df_train = df_train[relevant_cols + ['cancellation_datetime']]
    df_test = df_test[relevant_cols]
    # Prepare the training data
    X_train, y_train = prepare_data(df_train, date_cols, cat_cols, num_cols)

    # Prepare the test data
    X_test, _ = prepare_data(df_test, date_cols, cat_cols, num_cols, is_train=False)

    return X_train, y_train, X_test

def predict_test_data(model, X_test, output_filename):
    # Make predictions on the test data using the trained model
    y_test_pred = model.predict(X_test)

    # Create a DataFrame with the 'id' and 'cancellation' columns and save it
    output_df = pd.DataFrame({
        'id': X_test['h_booking_id'],
        'cancellation': y_test_pred
    })
    output_df.to_csv(output_filename, index=False)

def prepare_data(df, date_cols, cat_cols, num_cols,is_train=True):
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    # calculate days_before
    df['days_before'] = (df['checkin_date'] - df['booking_datetime']).dt.days

    # calculate num_of_days
    df['num_of_days'] = (df['checkout_date'] - df['checkin_date']).dt.days

    df = df.drop(['booking_datetime', 'checkin_date', 'checkout_date'], axis=1)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[col + '_year'] = df[col].dt.year
        # df[col + '_month'] = df[col].dt.month
        # df[col + '_day'] = df[col].dt.day
        # df[col + '_dayofweek'] = df[col].dt.dayofweek
    df = df.drop(date_cols, axis=1)

    if is_train:
        df['cancellation'] = df['cancellation_datetime'].notnull().astype(int)
        df = df.drop('cancellation_datetime', axis=1)

        return  df[num_cols + cat_cols ], df[['cancellation']]
    else:
        return df[num_cols + cat_cols], None


def prepare_pipeline(cat_cols, num_cols):



    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

    return preprocessor

def evaluate_model(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    print("Validation Accuracy:", accuracy)
    print("Validation Precision:", precision)
    print("Validation Recall:", recall)
    print("Validation F1 Score:", f1)

def main():
    df_train = pd.read_csv('agoda_cancellation_train.csv')
    # df_train = df_train.sample(n=10000, random_state=42)

    df_test = pd.read_csv('Agoda_Test_1.csv') # change to your test dataset
    # df_test = df_test.sample(n=2000, random_state=42)
    relevant_cols = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_country_code',
                     'hotel_star_rating', 'charge_option', 'accommadation_type_name',
                     'customer_nationality', 'guest_is_not_the_customer',
                     'no_of_room',
                     'original_payment_currency', 'is_first_booking', 'request_airport',
                     'hotel_brand_code', 'hotel_live_date', 'no_of_adults',
                     'no_of_children', 'is_user_logged_in','h_booking_id']

    date_cols = ['hotel_live_date']
    cat_cols = ['hotel_country_code', 'charge_option', 'accommadation_type_name', 'customer_nationality',
                 'original_payment_currency',
                'hotel_brand_code']
    num_cols = ['h_booking_id', 'hotel_star_rating', 'guest_is_not_the_customer', 'no_of_room', 'no_of_adults', 'no_of_children',
                'is_user_logged_in',  'days_before', 'num_of_days', 'hotel_live_date_year']


    X_train, y_train, X_test = preprocess_data(df_train, df_test, date_cols, cat_cols, num_cols,relevant_cols)

    # X_test = df_test.drop('cancellation', axis=1)
    # y_test = df_test['cancellation']

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    preprocessor = prepare_pipeline(cat_cols, num_cols)

    model = LGBMClassifier(random_state=42)

    pipeline = imbPipeline(steps=[('preprocessor', preprocessor),
                                  ('smote', SMOTE(random_state=42)),
                                  ('model', model)])

    param_grid = {
        'model__max_depth': [3, 7],
        'model__n_estimators': [50, 200],
        'model__learning_rate': [0.01, 0.2]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    evaluate_model(grid_search, X_val, y_val)

    # # Evaluate the model
    # evaluate_model(grid_search, X_val, y_val)

    # Predict on the test data and save the output
    predict_test_data(grid_search, X_test, 'agoda_cancellation_prediction.csv')

if __name__ == '__main__':
    main()
