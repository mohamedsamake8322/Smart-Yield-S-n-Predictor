# === evaluate_model.py ===

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import seaborn as sns
import joblib
import os

# Load data
df = pd.read_csv("data.csv")

# Features and target
features = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]
target = "Yield"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "LinearRegression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
}

# Directory to save results
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save model
    joblib.dump(model, f"models/{name.lower()}.pkl")

    # Evaluation
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ {name} Evaluation:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.2f}")

    # Visualization
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"{name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models/{name}_prediction_plot.png")
    plt.close()

print("\n📦 Models and plots saved in /models/")
