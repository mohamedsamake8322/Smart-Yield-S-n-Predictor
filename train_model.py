import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ✅ Device definition (CPU only)
device = torch.device("cpu")

# ✅ Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📂 Verify and create the "model" directory
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
logging.info(f"✅ Directory verified: {MODEL_DIR}")

# 📥 Automatic detection of dataset columns
def detect_input_size(csv_path="data.csv"):
    df = pd.read_csv(csv_path)
    logging.info(f"🔎 Available columns in the dataset: {df.columns.tolist()}")

    if "yield" not in df.columns:
        raise KeyError("🛑 Error: The 'yield' column does not exist in the dataset. Check your CSV file.")

    input_size = len(df.columns) - 1

    try:
        input_size = int(input_size)  # ✅ Convert to integer to avoid error
    except ValueError:
        logging.error(f"🛑 Error: `input_size` must be an integer, but received {type(input_size)}")
        raise TypeError(f"input_size must be an integer, but got {type(input_size)}")

    logging.info(f"✅ Feature detection: {input_size} columns used for prediction.")
    return input_size, df

# 📥 Load and preprocess data
def load_data(df):
    logging.info("🔄 Preprocessing the dataset...")

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    X = df.drop(columns=["yield"])
    y = df["yield"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 Define the PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32).to(device)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1).to(device)

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 📌 Save and load the model
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pth")

def save_model(model, path=MODEL_PATH):
    logging.info(f"🔍 Saved model keys: {model.state_dict().keys()}")
    torch.save(model.state_dict(), MODEL_PATH)  # ✅ Expected format
    
    # 🚨 Verification after saving
    if os.path.exists(path):
        logging.info(f"✅ Model correctly saved at {path}!")
    else:
        logging.error(f"🛑 Error: The model was not saved correctly.")
        raise RuntimeError("🛑 Model save failure.")

# 🚀 Function to train the model
def train_model():
    logging.info("🚀 Starting model training...")

    input_size, df = detect_input_size()

    if not isinstance(input_size, int):
        logging.error(f"🛑 `input_size` must be an integer, but received {type(input_size)} with value `{input_size}`")
        raise TypeError(f"`input_size` must be an integer, but got {type(input_size)}")

    X_train, X_test, y_train, y_test = load_data(df)
    model = PyTorchModel(input_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    for epoch in range(1000):
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    save_model(model)

    # 🚨 Final check to ensure model is properly saved
    if not os.path.exists(MODEL_PATH):
        logging.error(f"🛑 Critical error: `disease_model.pth` was not saved properly.")
        raise RuntimeError("🛑 The model was not saved despite training.")

    return model

# 🚀 End of training
logging.info("🎯 ✅ Model ready for use!")
if __name__ == "__main__":
    train_model()
