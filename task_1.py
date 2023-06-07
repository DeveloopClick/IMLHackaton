import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the training dataset
df_train = pd.read_csv('agoda_cancellation_train.csv')

# Drop rows with NaN values in 'cancellation_datetime'
df_train = df_train.dropna(subset=['cancellation_datetime'])

# Split the data into features (X) and target (y)
X = df_train.drop(['cancellation_datetime'], axis=1)
y = df_train['cancellation_datetime'].notnull().astype(int)

# Preprocess the data
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.6, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy)

# Load the test dataset
df_test = pd.read_csv('Agoda_Test_1.csv')

# Preprocess the test data
X_test = preprocessor.transform(df_test)

# Make predictions on the test data
y_test_pred = model.predict(X_test)

# Create a DataFrame with the 'id' and 'cancellation' columns and save it
output_df = pd.DataFrame({
    'id': df_test['h_booking_id'],
    'cancellation': y_test_pred
})
output_df.to_csv('agoda_cancellation_prediction.csv', index=False)
