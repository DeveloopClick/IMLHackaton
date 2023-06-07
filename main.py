import pandas as pd


def prepare_data(df, is_train=True):
    # Fill NaN values with defaults
    relevant_cols = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_country_code',
                     'hotel_star_rating', 'charge_option', 'accommadation_type_name',
                     'customer_nationality', 'guest_nationality_country_name',
                     'guest_is_not_the_customer', 'no_of_room', 'origin_country_code',
                     'original_payment_currency', 'is_first_booking', 'request_airport',
                     'hotel_brand_code', 'hotel_chain_code', 'hotel_live_date', 'no_of_adults',
                     'no_of_children', 'is_user_logged_in']

    # Fill NaN values with defaults
    default_values = {
        'hotel_country_code': 'other',
        'hotel_star_rating': 0.0,
        'charge_option': 'other',
        'accommadation_type_name': 'other',
        'customer_nationality': 'other',
        'guest_nationality_country_name': 'other',
        'guest_is_not_the_customer': 0,
        'no_of_room': 0,
        'origin_country_code': 'other',
        'original_payment_currency': 'other',
        'is_first_booking': False,
        'request_airport': 0,
        'hotel_brand_code': 'other',
        'hotel_chain_code': 'other',
        'no_of_adults': 0,
        'no_of_children': 0,
        'is_user_logged_in': False
    }
    df = df.fillna(default_values)

    # Prepare the data
    X = df[relevant_cols].copy()
    if is_train:
        X = X.dropna(subset=['cancellation_datetime'])
        y = X['cancellation_datetime'].notnull().astype(int)
    else:
        y = None

    # Convert date and time columns to numerical representation
    date_cols = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date']
    for col in date_cols:
        X[col] = (pd.to_datetime(X[col]).astype('int64') // 10**9).astype('int32')

    # Convert categorical columns to strings
    cat_cols = ['hotel_country_code', 'accommadation_type_name',
                'customer_nationality', 'guest_nationality_country_name',
                'origin_country_code', 'original_payment_currency',
                'hotel_brand_code', 'hotel_chain_code']

    for col in cat_cols:
        X[col] = X[col].astype(str)

    # Apply label encoders
    label_encoders = {}
    for col in cat_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    return X, y
