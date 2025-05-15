import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Check if the dataset exists
if not os.path.exists("data.csv"):
    raise FileNotFoundError("❌ data.csv not found. Please check its location.")

print("🔄 Loading dataset...")

# Load the data
df = pd.read_csv("data.csv")

# Encode categorical columns
df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])

# Separate features and target variable
X = df_encoded.drop("yield", axis=1)
y = df_encoded["yield"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("🚀 Training the model...")
# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully. RMSE: {rmse:.2f}, R2: {r2:.2f}")

# Save the trained model
joblib.dump(model, "model_xgb.pkl", compress=3)
joblib.dump(model, "yield_model.pkl", compress=3)

print("✅ Model saved as model_xgb.pkl and yield_model.pkl")