# 🏛️ AI Smart Real Estate Price Prediction & Valuation System

An institutional-grade, AI-powered residential property valuation engine and geospatial intelligence platform engineered for Bengaluru's dynamic real estate market.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Framework](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![Machine Learning](https://img.shields.io/badge/CatBoost-Regressor-yellow)
![Maps](https://img.shields.io/badge/Leaflet%20%7C%20OpenStreetMap-Tiles-orange?logo=openstreetmap)
![Status](https://img.shields.io/badge/Status-Production%20Ready-emerald)

---

## ✨ Features

- **Hybrid AI Valuation Engine:** Combines localized micro-market pricing (65% baseline) with a multi-feature CatBoost gradient-boosted regressor (35% regression) trained on 13.3K+ historical listings with an **82.4% $R^2$ model accuracy**.
- **Interactive Geospatial Intelligence:** 
  - Point-and-click interactive mapping with OpenStreetMap, Esri Photoreal Satellite, and Topographic Elevation views.
  - 1km micro-market catchment circle & radius analysis.
  - Drop-pin mode, GPS locator, and quick-select chips for prime Bengaluru hubs (Indiranagar, Koramangala, Whitefield, HSR Layout, Electronic City, Bellandur, Hebbal, etc.).
- **Smart Natural Language AI Concierge:** 
  - Built-in conversational real estate consultant. Ask queries like *"Find 3BHK in Indiranagar under 2 Cr"* or *"Compare Whitefield vs HSR Layout"*.
  - 1-click **"Apply to Valuation Engine"** that synchronizes parameters directly to the map and form.
- **Explainable Valuation Dossier:**
  - Fair market valuation (₹ Crores & Lakhs) with a $\pm 5\%$ algorithmic confidence volatility band.
  - Micro-market demand momentum score (1–10), yield/investment index, and social infrastructure scores (CBD distance, schools, hospitals).
- **Dynamic Financing & Mortgage Simulator:**
  - Real-time loan tenure, interest rate, and down payment sliders.
  - Dynamic **Karnataka Stamp Duty (5.6%) & Registration (1.0%)** toggle for accurate total acquisition cost planning.
- **Bank-Ready PDF Export:** One-click executive dossier generation formatted for print and bank appraisal.

---

## 🏗️ Tech Stack

- **Backend:** Python, Flask, Pandas, NumPy, CatBoost
- **Frontend:** HTML5, CSS3 (Modern Glassmorphism & Executive Dark/Gold Theme), JavaScript (ES6+)
- **Mapping & GIS:** Leaflet.js, OpenStreetMap (Free, No API key required), Google Maps JS API (optional hybrid fallback), Nominatim Geocoding
- **Machine Learning Models:** `CatBoostRegressor` serialized via `pickle`, locality price benchmarks, spatial proximity matrices

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Jayant-Hu/AI-Smart-Real-Estate-Price-Prediction-System.git
cd AI-Smart-Real-Estate-Price-Prediction-System
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or install core packages: `pip install flask pandas numpy catboost scikit-learn`)*

### 4. Run the Application
```bash
python backend/main.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py                 # Flask server, ML inference pipeline & AI concierge API
│   ├── static/
│   │   ├── style.css           # Luxury dark slate & gold design system
│   │   └── script.js           # Supplementary client script
│   └── templates/
│       ├── index.html          # Interactive map workspace & specifications form
│       └── result.html         # Valuation dossier & mortgage breakdown
├── data/
│   ├── raw/bangalore.csv       # Raw listing dataset
│   ├── cleaned_data.csv        # Preprocessed features
│   └── geo_data.csv            # Geocoded micro-market coordinates
├── ml/
│   ├── data_cleaning.py        # Outlier removal & data sanitization
│   ├── geo_features.py         # Proximity & density calculations
│   └── train_model.py          # CatBoost model training script
├── models/
│   ├── house_price_model.pkl   # Serialized CatBoost ML model
│   ├── columns.pkl             # Feature column map
│   ├── location_price.pkl      # Location lookup map
│   └── location_price_map.csv  # Micro-market baseline rates
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 📄 License
This project is licensed under the MIT License.
