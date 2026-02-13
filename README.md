# 🌬️ AQI Sentinel  
## AI-Powered Real-Time Air Quality Intelligence & Forecasting Platform for India

AQI Sentinel is an advanced environmental intelligence system designed to monitor, analyze, forecast, and explain air pollution trends across India.  

The platform combines Deep Learning (LSTM-based time-series modeling) with Explainable AI (SHAP) to provide not only accurate PM2.5 predictions but also interpretable insights into pollutant behavior and AQI fluctuations.

Built using Python, TensorFlow, Flask, and interactive visualization tools, AQI Sentinel functions as a scalable decision-support system for environmental analytics.

---

# 🌍 Problem Statement

India faces severe air pollution challenges due to:

- Rapid urbanization  
- Industrial emissions  
- Vehicular pollution  
- Seasonal crop burning  
- Construction dust and urban heat effects  

While AQI data is publicly available, most systems:

- Only display current AQI  
- Do not provide predictive insights  
- Lack transparency in model reasoning  
- Offer limited analytical capabilities  

AQI Sentinel addresses these gaps by introducing predictive intelligence and explainable AI into air quality monitoring.

---

# 🎯 Objectives

The primary objectives of AQI Sentinel are:

- Monitor real-time air pollution metrics nationwide  
- Forecast next-day PM2.5 concentrations using deep learning  
- Identify key pollutant drivers influencing AQI changes  
- Provide interactive geospatial and statistical visualizations  
- Enable data-driven environmental decision-making  

---

# 🏗️ System Architecture

CPCB CAAQMS Data Source
↓
Data Ingestion & Cleaning
↓
Feature Engineering & Scaling
↓
LSTM Time-Series Forecasting Model
↓
SHAP Explainability Layer
↓
Flask REST API
↓
Interactive Web Dashboard


---

# 🔄 Data Pipeline

## 1️⃣ Data Collection
- Real-time data from 300+ CPCB CAAQMS stations
- Historical multi-year datasets for model training
- Pollutants monitored:
  - PM2.5
  - PM10
  - NO2
  - SO2
  - CO
  - O3

## 2️⃣ Data Preprocessing
- Missing value handling
- Outlier detection
- Feature normalization
- Time-series windowing
- Lag feature creation

## 3️⃣ Feature Engineering
- Rolling averages
- Temporal features (hour, day, season)
- Pollutant interaction patterns

---

# 📈 Deep Learning Model

## LSTM Architecture

AQI Sentinel uses a Long Short-Term Memory (LSTM) neural network to capture long-term temporal dependencies in pollution data.

### Why LSTM?

- Handles sequential time-series data
- Learns seasonal pollution patterns
- Captures temporal pollutant correlations
- Reduces vanishing gradient issues

### Model Configuration

- Input: Multivariate time-series pollutant features
- Layers:
  - LSTM layers
  - Dropout for regularization
  - Dense output layer
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

---

# 🧠 Explainable AI (SHAP Integration)

Most deep learning models operate as black boxes. AQI Sentinel integrates SHAP (SHapley Additive exPlanations) to provide interpretability.

SHAP enables:

- Feature importance visualization
- Pollutant contribution analysis
- Model transparency
- Trustworthy forecasting

Example Insight:
- Increased NO2 → Higher predicted PM2.5 (traffic influence)
- Elevated SO2 → Industrial emission impact

---

# 🗺️ Interactive Dashboard

The dashboard provides:

## 🌍 Geospatial Visualization
- Interactive Folium map
- AQI color-coded station markers
- City-level air quality insights

## 📊 Analytical Charts
- Historical pollutant trends
- Forecast vs actual comparison
- Pollutant distribution analysis
- Time-series behavior plots

## 📱 Responsive UI
- Desktop optimized
- Clean layout
- Real-time data rendering

---

# 🧪 Model Evaluation

The model is evaluated using:

- Root Mean Square Error (RMSE)
- R² Score
- Forecast accuracy comparison
- Multi-city validation

Training data includes major Indian metros such as:

- Delhi
- Mumbai
- Chennai
- Bengaluru
- Kolkata

Optimization Techniques:

- Dropout layers to prevent overfitting
- Data normalization
- Temporal window tuning
- Learning rate optimization

---

# 🧰 Tech Stack

## Programming
- Python 3.x

## Data Processing
- Pandas
- NumPy

## Machine Learning
- TensorFlow
- Keras (LSTM)
- Scikit-learn

## Explainability
- SHAP

## Backend
- Flask (REST APIs)

## Visualization
- Folium
- Plotly
- Matplotlib

## Tools
- Git
- Jupyter Notebook

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

git clone [https://github.com/YOUR_USERNAME/AQI-Sentinel.git](https://github.com/SUDHARSHAN-15/AIR-QUALITY-ANALYSIS)
cd AQI-Sentinel


## 2️⃣ Install Dependencies

pip install -r requirements.txt


## 3️⃣ Train the Model

python train_lstm.py


## 4️⃣ Generate SHAP Explanations

python explain_shap.py


## 5️⃣ Run the Web Application

python app.py


---

# 📂 Data Source

Central Pollution Control Board (CPCB)  
Continuous Ambient Air Quality Monitoring Stations (CAAQMS)  

Official Station List:  
https://cpcb.nic.in/upload/national-air-quality-index/Station_List_Of_CAAQMS.pdf  

---

# 🚀 Future Enhancements

- Integration with weather APIs
- Multi-step forecasting (7-day prediction)
- Cloud deployment (AWS / Azure)
- Real-time alert notifications
- Mobile application integration
- Auto-scaling model retraining pipeline

---

# 🌱 Real-World Impact

AQI Sentinel can support:

- Urban environmental planning
- Smart city initiatives
- Industrial emission monitoring
- Public health awareness
- Policy-driven pollution control strategies

---

# 👤 Author

**Sudharshan M**  
 

---


# ⭐ If you found this project insightful, consider giving it a star.
