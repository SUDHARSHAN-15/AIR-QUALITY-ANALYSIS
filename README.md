# AQI Sentinel  
**AI-Powered Real-Time Air Quality Guardian for India**  
300+ CPCB CAAQMS Stations • LSTM Forecasting • SHAP Explainability • Interactive Dashboard

![AQI Sentinel](https://via.placeholder.com/1200x400.png?text=AQI+Sentinel+Project+Banner)

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Flask](https://img.shields.io/badge/Flask-2.0-green.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)]()

---

## 📌 Overview
**AQI Sentinel** is a nationwide AI-powered system designed to monitor, analyze, and forecast India’s air quality. It processes real-time data from **300+ CPCB CAAQMS stations**, applies **LSTM deep learning models** for PM2.5 prediction, and uses **SHAP explainability** to interpret model decisions.  
The project includes an interactive map, analytics dashboard, forecasts, and pollutant breakdowns.

---

## ✨ Key Features
- 🌐 Real-time data from **300+ monitoring stations**  
- 📈 LSTM-based **next-day PM2.5 forecasting**  
- 🧠 **SHAP explainability** for understanding predictions  
- 🗺️ Interactive **Folium map** with AQI-coded markers  
- 📊 Dynamic charts (Plotly & Matplotlib)  
- 📱 Fully responsive UI  
- 🚀 Easy to deploy on PythonAnywhere / Render  

---

## 📂 Data Source
**CPCB – Continuous Ambient Air Quality Monitoring Stations (CAAQMS)**  
Updated: *25-03-2019*  
Full station list: https://cpcb.nic.in/upload/national-air-quality-index/Station_List_Of_CAAQMS.pdf

---

## 🧠 Tech Stack
- **Backend:** Flask  
- **ML Model:** TensorFlow LSTM  
- **Explainability:** SHAP  
- **Visualization:** Folium, Plotly, Matplotlib  
- **Data:** CPCB (real-time) + Kaggle (historical)  

---

## 🚀 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/AQI-Sentinel.git
cd AQI-Sentinel
pip install -r requirements.txt
python train_lstm.py
python explain_shap.py
python app.py
