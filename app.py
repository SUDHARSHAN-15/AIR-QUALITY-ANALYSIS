# app.py - INDIA AIR GUARDIAN v3.0 | CPCB 2019 | 300+ Stations | All-City LSTM + SHAP
from flask import Flask, render_template, jsonify, request
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
import json
from datetime import datetime

app = Flask(__name__)

# === PATHS ===
DATA = Path("data")
PROCESSED = DATA / "processed"
CLUSTERED = DATA / "clustered"
MODEL = Path("models")
STATIONS_FILE = Path("stations.json")
STATIC = Path("static")

# === LOAD DATA ===
hourly = pd.read_csv(PROCESSED / "india_hourly.csv")
daily = pd.read_csv(PROCESSED / "india_daily.csv")
clusters = pd.read_csv(CLUSTERED / "india_city_clusters.csv")

# === LOAD MODEL & SCALERS (Per-City Scalers) ===
model = load_model(MODEL / "lstm_pm25_anycity.h5", compile=False)
scalers = joblib.load(MODEL / "scalers_anycity.pkl")  # Dictionary: {city: scaler}

# === LOAD FULL 300+ CPCB STATIONS FROM YOUR IMAGE ===
with open(STATIONS_FILE, "r", encoding="utf-8") as f:
    STATIONS = json.load(f)

# Cache predictions
pred_cache = {}

def predict_next_day(city):
    if city not in scalers:
        return None
    if city in pred_cache:
        return pred_cache[city]
    
    city_data = daily[daily['City'] == city].dropna(subset=['PM2.5'])
    if len(city_data) < 8:
        pred_cache[city] = None
        return None
    
    last7 = city_data['PM2.5'].tail(7).values.reshape(-1, 1)
    scaler = scalers[city]
    seq = scaler.transform(last7).reshape(1, 7, 1)
    pred = model.predict(seq, verbose=0)[0][0]
    result = round(float(scaler.inverse_transform([[pred]])[0][0]), 1)
    pred_cache[city] = result
    return result

@app.route('/')
def dashboard():
    # Dark India Map
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        attr="CPCB | OpenStreetMap"
    )
    mc = MarkerCluster(name="300+ CPCB Stations").add_to(m)

    # Add 300+ Real Stations + City AQI
    for city, info in STATIONS.items():
        # === CLUSTER COLOR ===
        city_cluster = clusters[clusters['City'] == city]
        cluster_id = int(city_cluster['cluster'].iloc[0]) if not city_cluster.empty else 1
        color = ['#00ff00', '#ffaa00', '#ff0000'][cluster_id]  # Green, Yellow, Red

        # === LATEST PM2.5 ===
        latest_pm25 = hourly[hourly['City'] == city]['PM2.5'].iloc[-1] if city in hourly['City'].values else 0
        pred = predict_next_day(city)
        pred_str = f"{pred}" if pred else "N/A"

        # === CITY MARKER (AQI + Forecast) ===
        folium.CircleMarker(
            location=[info['lat'], info['lon']],
            radius=12,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"""
            <div style="font-size:0.9rem; min-width:180px;">
                <b>{city}</b><br>
                Latest PM2.5: <b>{latest_pm25:.1f}</b> µg/m³<br>
                Tomorrow: <b>{pred_str}</b> µg/m³<br>
                <small>Cluster: {['Low', 'Medium', 'High'][cluster_id]}</small>
            </div>
            """,
            tooltip=f"{city}: {latest_pm25:.0f} → {pred_str}"
        ).add_to(mc)

        # === INDIVIDUAL 300+ CPCB STATIONS ===
        for station in info.get("stations", []):
            folium.CircleMarker(
                location=[station["lat"], station["lon"]],
                radius=6,
                color="#ff0066",
                fill=True,
                fill_color="#ff0066",
                popup=f"""
                <div style="font-size:0.85rem;">
                    <b>{station['name'].split(',')[0]}</b><br>
                    <small>{city}, {info.get('state', '')}</small><br>
                    <i>CPCB CAAQMS Station</i>
                </div>
                """,
                tooltip=station["name"].split(",")[0]
            ).add_to(mc)

    map_html = m._repr_html_()

    # === TOP 5 POLLUTED ===
    top5 = hourly.groupby('City')['PM2.5'].last().sort_values(ascending=False).head(5)

    # === SHAP-READY CITIES (only those with scaler & SHAP plot) ===
    shap_cities = []
    for city in sorted(scalers.keys()):
        shap_file = STATIC / f"shap_{city.lower().replace(' ', '_')}.png"
        if shap_file.exists():
            shap_cities.append(city)

    return render_template(
        'index.html',
        map_html=map_html,
        top5=top5.to_dict(),
        total_cities=len(STATIONS),
        timestamp=datetime.now().strftime("%d %b %Y, %I:%M %p"),
        shap_cities=shap_cities
    )

@app.route('/api/predict')
def api_predict():
    city = request.args.get('city')
    if not city or city not in STATIONS:
        return jsonify({"error": "City not found"}), 400
    pred = predict_next_day(city)
    latest = hourly[hourly['City'] == city]['PM2.5'].iloc[-1] if city in hourly['City'].values else None
    return jsonify({
        "city": city,
        "latest": round(float(latest), 1) if latest else None,
        "predicted": pred
    })

if __name__ == '__main__':
    print("AQI SENTINEL v1.0 | 300+ CPCB Stations | AI + SHAP | LIVE")
    app.run(host='0.0.0.0', port=5000, debug=False)