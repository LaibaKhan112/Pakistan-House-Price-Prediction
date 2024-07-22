from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib

# Load the cleaned dataset
df = pd.read_csv('Cleaned_House.csv')
df["Area sqft"] = df["Area Size"] * 225
df = df.drop(columns=["Unnamed: 0", "latitude", "longitude", "Area Type", "Area Size"], axis=1)
print(df.head())

# Separate features (X) and target (y)
X = df.drop(columns=["price"])
y = df["price"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column transformer with OneHotEncoder for categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=False), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create a pipeline with preprocessing and model
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', LinearRegression())])

# Fit pipeline on training data
pipe.fit(X_train, y_train)

# Predict on test data
y_pred = pipe.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Evaluate the model's accuracy using R-squared
accuracy = r2_score(y_test, y_pred)
print(f"Accuracy (R-squared): {accuracy:.2f}")

# Save the trained model
joblib.dump(pipe, 'house_price_predictor.sav')
