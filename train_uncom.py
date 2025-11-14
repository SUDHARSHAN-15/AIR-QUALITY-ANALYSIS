# train_lstm_india.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

DATA  = Path("data/processed")
MODEL = Path("models")
MODEL.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Choose a demo city (change to any city you like)
# -------------------------------------------------
CITY = "Delhi"

df = pd.read_csv(DATA / "india_daily.csv")
city_df = df[df['City'] == CITY].dropna(subset=['PM2.5'])

if len(city_df) < 30:
    raise ValueError(f"Not enough data for {CITY}")

series = city_df['PM2.5'].values.reshape(-1, 1)

scaler = MinMaxScaler()
series_s = scaler.fit_transform(series)

SEQ_LEN = 7
def create_sequences(data, n):
    X, y = [], []
    for i in range(len(data) - n):
        X.append(data[i:i + n])
        y.append(data[i + n])
    return np.array(X), np.array(y)

X, y = create_sequences(series_s, SEQ_LEN)
X = X.reshape((X.shape[0], SEQ_LEN, 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---- Build model (NO compile yet) ----
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)),
    Dense(1)
])

# Compile **only for training**
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train,
          epochs=30, batch_size=32,
          validation_split=0.1, verbose=1)

# ---- Save UN-COMPILED model + scaler + city name ----
model.save(MODEL / "lstm_pm25_anycity.h5", include_optimizer=False)
joblib.dump(scaler, MODEL / "scaler_pm25_anycity.pkl")
joblib.dump(CITY,   MODEL / "demo_city.pkl")

print(f"\nModel trained on **{CITY}** and saved (un-compiled).")