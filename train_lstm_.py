# train_lstm.py - TRAIN ONE LSTM ON ALL CITIES (FIXED)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from pathlib import Path

DATA = Path("data/processed")
MODEL = Path("models")
MODEL.mkdir(exist_ok=True)

# Load all daily data
df = pd.read_csv(DATA / "india_daily.csv")
df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])  # FIXED: use correct column
df = df.sort_values(['City', 'datetime_hour'])

print(f"Loaded {len(df)} daily records from {df['City'].nunique()} cities")

# Prepare sequences for ALL cities
def create_sequences(city_data, seq_len=7):
    scaler = MinMaxScaler()
    values = city_data['PM2.5'].dropna().values
    if len(values) < seq_len + 1:
        return None, None, None
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    return np.array(X), np.array(y), scaler

X_all, y_all = [], []
scalers = {}

for city in df['City'].unique():
    city_df = df[df['City'] == city]
    X, y, scaler = create_sequences(city_df)
    if X is not None and len(X) > 0:
        X_all.append(X)
        y_all.append(y)
        scalers[city] = scaler
        print(f"Added {len(X)} sequences for {city}")

if not X_all:
    raise ValueError("No city has enough data!")

X_all = np.vstack(X_all)
y_all = np.vstack(y_all)

print(f"Training on {X_all.shape[0]} sequences from {len(scalers)} cities")

# Build LSTM (more powerful)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(7, 1)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train with early stopping
early_stop = EarlyStopping(patience=15, restore_best_weights=True)
model.fit(X_all, y_all, epochs=100, batch_size=64, 
          validation_split=0.2, callbacks=[early_stop], verbose=1)

# Save
model.save(MODEL / "lstm_pm25_anycity.h5")
joblib.dump(scalers, MODEL / "scalers_anycity.pkl")
print(f"✅ Model trained on {len(scalers)} cities!")
print(f"✅ Scalers saved for each city")
print("✅ Ready for SHAP explanations")