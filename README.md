AQI Sentinel
AI-Powered Real-Time Air Quality Monitoring & Forecasting System for India

Tech Stack: Python • Flask • TensorFlow (LSTM) • SHAP • Folium • Plotly • Matplotlib
Data Sources: CPCB CAAQMS (Real-time) • Historical Air Quality Datasets

📌 Project Overview

AQI Sentinel is an AI-driven air quality monitoring and forecasting platform designed to analyze and predict air pollution trends across India. The system processes real-time data from 300+ Continuous Ambient Air Quality Monitoring Stations (CAAQMS) maintained by the Central Pollution Control Board (CPCB).

The platform integrates deep learning models for PM2.5 prediction, explainable AI techniques for interpretability, and an interactive analytics dashboard for visualization and decision support.

This project demonstrates expertise in:

Machine Learning & Deep Learning

Time-Series Forecasting (LSTM)

Explainable AI (SHAP)

REST-based Web Applications (Flask)

Data Visualization & Dashboarding

🎯 Key Objectives

Monitor nationwide air quality metrics in real time

Forecast next-day PM2.5 levels using deep learning

Provide explainable predictions using SHAP values

Enable interactive exploration of pollution data through dashboards

Support data-driven environmental awareness and decision-making

✨ Core Features
🌐 Real-Time Air Quality Monitoring

Integrates data from 300+ CPCB CAAQMS stations

Displays pollutant metrics including PM2.5, PM10, NO₂, SO₂, CO, O₃

AQI-based color-coded visualization

📈 LSTM-Based Forecasting

Time-series forecasting using TensorFlow LSTM

Predicts next-day PM2.5 concentration

Optimized using historical pollutant trends

🧠 Explainable AI with SHAP

SHAP-based feature importance analysis

Identifies key pollutant drivers influencing predictions

Enhances transparency and model interpretability

🗺️ Interactive Visualization Dashboard

Folium-based geospatial AQI map

Dynamic charts using Plotly & Matplotlib

Pollutant trend comparison across cities

Responsive UI for desktop and mobile

📊 System Architecture

Data Ingestion

Real-time CPCB station data

Historical datasets for training

Data Processing

Cleaning, normalization, and feature engineering

Time-series structuring

Model Training

LSTM network for PM2.5 forecasting

Evaluation using regression metrics

Explainability Layer

SHAP value computation

Feature impact visualization

Web Deployment

Flask backend

Interactive frontend dashboard

📂 Data Source

Central Pollution Control Board (CPCB)
Continuous Ambient Air Quality Monitoring Stations (CAAQMS)
Station List (Reference):
https://cpcb.nic.in/upload/national-air-quality-index/Station_List_Of_CAAQMS.pdf

🧠 Technologies Used
Programming

Python (Pandas, NumPy)

Machine Learning & Deep Learning

TensorFlow (LSTM)

Scikit-learn

SHAP

Backend

Flask (REST APIs)

Visualization

Folium (Geospatial Mapping)

Plotly

Matplotlib

Tools

Git

Jupyter Notebook

📈 Model Performance

Time-series forecasting using LSTM architecture

Optimized using historical pollutant patterns

Evaluated using RMSE and R² metrics

(You can optionally add exact metrics here if available.)

🚀 Run Locally
git clone https://github.com/YOUR_USERNAME/AQI-Sentinel.git
cd AQI-Sentinel

pip install -r requirements.txt

# Train Model
python train_lstm.py

# Generate SHAP Explainability
python explain_shap.py

# Run Application
python app.py

💡 Deployment

The application can be deployed on:

PythonAnywhere

Render

Any cloud platform supporting Flask applications

👤 Author

Sudharshan M
