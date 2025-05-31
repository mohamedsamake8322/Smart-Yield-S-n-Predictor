import torch
import torch.nn as nn
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

# ✅ Define device (CPU only)
device = torch.device("cpu")

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Automatic column detection ----------
def detect_input_size(csv_path="data.csv"):
    df = pd.read_csv(csv_path)
    logging.info(f"🔎 Available columns in the dataset: {df.columns.tolist()}")

    if "yield" not in df.columns:
        raise KeyError("🛑 Error: The 'yield' column does not exist in the dataset. Check your CSV file.")

    input_size = len(df.columns) - 1

    try:
        input_size = int(input_size)  # ✅ Convert to integer to avoid errors
    except ValueError:
        logging.error(f"🛑 Error: `input_size` must be an integer, but received {type(input_size)}")
        raise TypeError(f"input_size must be an integer, but got {type(input_size)}")

    logging.info(f"✅ Feature detection: {input_size} columns used for prediction.")
    return input_size, df


# ---------- PyTorch model definition ----------
class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64).to(device)
        self.batch_norm1 = nn.BatchNorm1d(64)  # ✅ Added BatchNorm
        self.fc2 = nn.Linear(64, 32).to(device)
        self.batch_norm2 = nn.BatchNorm1d(32)  # ✅ Added BatchNorm
        self.fc3 = nn.Linear(32, 1).to(device)

        self.activation = nn.LeakyReLU(negative_slope=0.01)  # ✅ Improved activation
        self.dropout = nn.Dropout(0.3)  # ✅ Regularization to prevent overfitting

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
