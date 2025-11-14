# explain_shap.py - SHAP FOR ALL CITIES (FINAL)
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import json

DATA = Path("data/processed")
MODEL = Path("models")
STATIC = Path("static")
STATIC.mkdir(exist_ok=True)

# Load model & scalers
model = load_model(MODEL / "lstm_pm25_anycity.h5", compile=False)
scalers = joblib.load(MODEL / "scalers_anycity.pkl")

# Load stations JSON
with open("stations.json", "r", encoding="utf-8") as f:
    STATIONS = json.load(f)

# Load data
daily = pd.read_csv(DATA / "india_daily.csv")

def explain_city(city):
    if city not in scalers:
        print(f"❌ No scaler for {city}")
        return False
    
    city_data = daily[daily['City'] == city].dropna(subset=['PM2.5']).tail(100)
    if len(city_data) < 10:
        print(f"❌ Not enough data for {city}")
        return False
    
    values = city_data['PM2.5'].values
    scaler = scalers[city]
    scaled = scaler.transform(values.reshape(-1, 1))

    # Create sequences
    seq_len = 7
    X = []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
    X = np.array(X)
    
    if len(X) < 10:
        print(f"❌ Not enough sequences for {city}")
        return False
    
    X_test = X[-10:]  # Last 10 days
    X_2d = X_test.reshape(X_test.shape[0], -1)  # (10, 7)

    # Background
    background = X[-50:].reshape(-1, seq_len)  # (50, 7)

    # Predict function
    def predict_fn(x):
        return model.predict(x.reshape(-1, seq_len, 1), verbose=0).flatten()

    print(f"Computing SHAP for {city}...")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_2d, nsamples=50)

    # Summary plot
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X_2d,
        feature_names=[f"Day-{i}" for i in range(seq_len, 0, -1)],
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP: PM2.5 Drivers in {city}")
    plt.tight_layout()
    plt.savefig(STATIC / f"shap_{city.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP saved: shap_{city.lower().replace(' ', '_')}.png")
    return True

# Explain ALL cities with data
successful = 0
for city in sorted(scalers.keys()):
    if explain_city(city):
        successful += 1

print(f"\n🎉 SHAP completed for {successful}/{len(scalers)} cities!")
print(f"📁 Plots saved in: static/")