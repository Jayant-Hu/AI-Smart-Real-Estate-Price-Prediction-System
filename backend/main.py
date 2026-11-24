from flask import Flask, render_template, request, jsonify

import pickle
import pandas as pd
import numpy as np
import math
import re

app = Flask(__name__)

# ---------------------------------------
# LOAD MODEL
# ---------------------------------------

model = pickle.load(
    open(
        "models/house_price_model.pkl",
        "rb"
    )
)

# ---------------------------------------
# LOAD COLUMNS
# ---------------------------------------

columns = pickle.load(
    open(
        "models/columns.pkl",
        "rb"
    )
)

# ---------------------------------------
# LOAD GEO DATA
# ---------------------------------------

geo_data = pd.read_csv(
    "data/geo_data.csv"
)

# Pre-aggregate unique locations for ultra-fast lookup
unique_geo = (
    geo_data.groupby('location')[['latitude', 'longitude']]
    .mean()
    .reset_index()
)

# ---------------------------------------
# LOAD LOCATION PRICE MAP
# ---------------------------------------

location_price_map_df = pd.read_csv(
    "models/location_price_map.csv"
)

location_price_map = dict(
    zip(
        location_price_map_df['location'],
        location_price_map_df['price_per_sqft']
    )
)

# Build serializable list of locations for frontend search
locations_list = [
    {
        "name": str(row['location']).strip(),
        "lat": round(float(row['latitude']), 6),
        "lon": round(float(row['longitude']), 6),
        "price_sqft": round(float(location_price_map.get(row['location'], 7500)), 1)
    }
    for _, row in unique_geo.iterrows()
    if pd.notna(row['location']) and pd.notna(row['latitude']) and pd.notna(row['longitude'])
]

# Sort alphabetically by name
locations_list.sort(key=lambda x: x['name'].lower())

# ---------------------------------------
# PREMIUM LOCATIONS
# ---------------------------------------

premium_locations = [
    'Indiranagar',
    'Koramangala',
    'Whitefield',
    'HSR Layout',
    'Electronic City',
    'Marathahalli',
    'Sarjapur Road',
    'JP Nagar',
    'Jayanagar'
]

# ---------------------------------------
# FIND NEAREST LOCATION (VECTORIZED)
# ---------------------------------------

def find_nearest_location(lat, lon):
    lats = unique_geo['latitude'].to_numpy()
    lons = unique_geo['longitude'].to_numpy()
    dists = (lats - lat)**2 + (lons - lon)**2
    nearest_idx = int(np.argmin(dists))
    return unique_geo.iloc[nearest_idx]['location']

# ---------------------------------------
# GEO FEATURES
# ---------------------------------------

def get_geo_features(lat, lon):

    center_lat = 12.9716
    center_lon = 77.5946

    # Distance
    distance_center = math.sqrt(
        (lat - center_lat)**2 +
        (lon - center_lon)**2
    )

    # Trend Score
    trend_score = 10 - (distance_center * 10)
    trend_score = max(1, min(10, trend_score))

    # Schools
    schools = max(1, int(10 - distance_center * 5))

    # Hospitals
    hospitals = max(1, int(6 - distance_center * 3))

    return (
        schools,
        hospitals,
        distance_center,
        trend_score
    )

# ---------------------------------------
# API ROUTES
# ---------------------------------------

import os
GOOGLE_MAPS_API_KEY = os.environ.get(
    "GOOGLE_MAPS_API_KEY",
    "AIzaSyAx45MvFENu_hnGgLy2lHQ2pW9z1WaTgjs"
)
CARTO_API_KEY = os.environ.get(
    "CARTO_API_KEY",
    "cb1_58k0_1_9411212b0c1588afc5b8d754"
)

@app.route('/api/locations')
def get_locations_api():
    return jsonify(locations_list)

# ---------------------------------------
# HOME PAGE
# ---------------------------------------

@app.route('/')
def home():
    return render_template(
        "index.html",
        locations=locations_list,
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
        carto_api_key=CARTO_API_KEY
    )

# ---------------------------------------
# PREDICTION
# ---------------------------------------

@app.route('/predict', methods=['POST'])

def predict():

    try:

        # ---------------------------------------
        # FORM / JSON VALUES
        # ---------------------------------------

        data = request.get_json(silent=True) or request.form

        lat = float(data['lat'])
        lon = float(data['lon'])
        sqft = float(data['total_sqft'])
        bath = float(data['bath'])
        bhk = float(data['bhk'])

        # ---------------------------------------
        # SAFETY FIX
        # ---------------------------------------

        if bhk == 0:
            bhk = 1

        # ---------------------------------------
        # FIND LOCATION
        # ---------------------------------------

        location = find_nearest_location(
            lat,
            lon
        )

        print(f"[Detected Location] {location}")

        # ---------------------------------------
        # GEO FEATURES
        # ---------------------------------------

        schools, hospitals, distance_center, trend_score = get_geo_features(
            lat,
            lon
        )

        # ---------------------------------------
        # AREA AVG PRICE
        # ---------------------------------------

        area_avg_price = location_price_map.get(
            location,
            8000
        )

        # ---------------------------------------
        # SMART FEATURES
        # ---------------------------------------

        luxury_score = sqft / bhk

        bath_per_bhk = bath / bhk

        estimated_pps = area_avg_price

        # ---------------------------------------
        # NEW ADVANCED FEATURES
        # ---------------------------------------

        location_density = 50

        premium_location = 1 if location in premium_locations else 0

        log_sqft = np.log(sqft)

        bhk_density = sqft / bhk

        bath_luxury_score = bath * bhk

        room_density = sqft / (bhk + bath)

        family_score = (
            (bhk * 2) + bath
        )

        investment_score = (
            trend_score *
            area_avg_price
        ) / 1000

        # ---------------------------------------
        # CREATE INPUT VECTOR
        # ---------------------------------------

        x = np.zeros(len(columns))

        # ---------------------------------------
        # FEATURE MAP
        # ---------------------------------------

        feature_map = {

            'total_sqft': sqft,

            'bath': bath,

            'bhk': bhk,

            'price_per_sqft': estimated_pps,

            'estimated_pps': estimated_pps,

            'luxury_score': luxury_score,

            'bath_per_bhk': bath_per_bhk,

            'latitude': lat,

            'longitude': lon,

            'area_avg_price': area_avg_price,

            'schools': schools,

            'hospitals': hospitals,

            'distance_center': distance_center,

            'trend_score': trend_score,

            'location_density': location_density,

            'premium_location': premium_location,

            'log_sqft': log_sqft,

            'bhk_density': bhk_density,

            'bath_luxury_score': bath_luxury_score,

            'room_density': room_density,

            'family_score': family_score,

            'investment_score': investment_score
        }

        # ---------------------------------------
        # INSERT FEATURES
        # ---------------------------------------

        for feature, value in feature_map.items():

            if feature in columns:

                idx = columns.index(feature)

                x[idx] = value

        # ---------------------------------------
        # LOCATION ENCODING
        # ---------------------------------------

        loc_col = f"location_{location}"

        if loc_col in columns:

            idx = columns.index(loc_col)

            x[idx] = 1

        # ---------------------------------------
        # ML PREDICTION
        # ---------------------------------------

        ml_prediction = model.predict([x])[0]

        # ---------------------------------------
        # MARKET BASE PRICE
        # ---------------------------------------

        base_price = (
            sqft *
            area_avg_price
        ) / 100000

        # ---------------------------------------
        # HYBRID AI PREDICTION
        # ---------------------------------------

        final_price = (
            (base_price * 0.65)
            +
            (ml_prediction * 0.35)
        )

        # ---------------------------------------
        # PREMIUM ADJUSTMENTS
        # ---------------------------------------

        if bhk >= 3:

            final_price *= 1.05

        if bath > bhk:

            final_price *= 1.03

        if sqft > 2500:

            final_price *= 1.08

        if premium_location == 1:

            final_price *= 1.12

        if trend_score >= 8:

            final_price *= 1.06

        # ---------------------------------------
        # ULTRA LUXURY BONUS
        # ---------------------------------------

        if sqft > 4000:

            final_price *= 1.15

        # ---------------------------------------
        # FINAL ROUND & ENRICHED METRICS
        # ---------------------------------------

        # Check if location is outside Bangalore Metropolitan Area
        # Greater Bengaluru typically falls within lat: 12.65 - 13.30, lon: 77.30 - 77.90
        is_out_of_range = not (12.65 <= lat <= 13.30 and 77.30 <= lon <= 77.90)

        final_price = round(final_price, 2)

        if is_out_of_range:
            price_crores = "Outside the range"
            price_range_min = "Outside the range"
            price_range_max = "Outside the range"
            estimated_emi = "N/A"
        else:
            if final_price >= 100:
                price_crores = f"₹ {final_price / 100:.2f} Cr"
            else:
                price_crores = f"₹ {final_price:.2f} Lakhs"

            price_range_min = round(final_price * 0.95, 2)
            price_range_max = round(final_price * 1.05, 2)

            # Standard estimated EMI for 80% loan, 8.5% interest, 20 years
            loan_principal = final_price * 100000 * 0.8
            monthly_r = 0.085 / 12
            months = 240
            if loan_principal > 0:
                emi_raw = loan_principal * monthly_r * ((1 + monthly_r)**months) / (((1 + monthly_r)**months) - 1)
                estimated_emi = f"{int(round(emi_raw)):,}"
            else:
                estimated_emi = "0"

        approx_dist_km = round(distance_center * 111.0, 1)

        context = {
            "prediction": final_price,
            "is_out_of_range": is_out_of_range,
            "price_crores": price_crores,
            "price_range_min": price_range_min,
            "price_range_max": price_range_max,
            "approx_dist_km": approx_dist_km,
            "estimated_emi": estimated_emi,
            "base_price": round(base_price, 2),
            "ml_prediction": round(ml_prediction, 2),
            "luxury_score": round(luxury_score, 1),
            "bath_per_bhk": round(bath_per_bhk, 2),
            "location": location,
            "sqft": int(sqft) if sqft.is_integer() else sqft,
            "bhk": int(bhk) if bhk.is_integer() else bhk,
            "bath": int(bath) if bath.is_integer() else bath,
            "schools": schools,
            "hospitals": hospitals,
            "trend_score": round(trend_score, 1),
            "distance_center": round(distance_center, 3),
            "price_sqft": round(area_avg_price, 1),
            "premium_location": premium_location,
            "investment_score": round(investment_score, 1),
            "lat": lat,
            "lon": lon,
            "google_maps_api_key": GOOGLE_MAPS_API_KEY,
            "carto_api_key": CARTO_API_KEY
        }

        # Support JSON clients or direct form post
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": True, **context})

        return render_template("result.html", **context)

    except Exception as e:
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": False, "error": str(e)}), 400
        return f"ERROR: {e}", 500

# ---------------------------------------
# JSON API PREDICT ROUTE
# ---------------------------------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return predict()

# ---------------------------------------
# AI CONCIERGE ASSISTANT ROUTE
# ---------------------------------------

@app.route('/api/ai-concierge', methods=['POST'])
def ai_concierge():
    try:
        data = request.get_json(silent=True) or {}
        query = str(data.get('query', '')).strip().lower()

        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400

        # 1. Match Locality (Robust Space/Punctuation Invariant Matching)
        matched_loc = None
        norm_query = re.sub(r'[^a-z0-9]', '', query)

        # Direct search with normalized strings
        for loc in locations_list:
            norm_loc = re.sub(r'[^a-z0-9]', '', loc['name'].lower())
            if len(norm_loc) >= 4 and norm_loc in norm_query:
                matched_loc = loc
                break

        # Check common aliases
        if not matched_loc:
            aliases = {
                "ecity": "Electronic City",
                "hsr": "HSR Layout",
                "jpnagar": "JP Nagar",
                "whitefield": "Whitefield",
                "marathahalli": "Marathahalli",
                "marathalli": "Marathahalli",
                "indiranagar": "Indira Nagar",
                "koramangala": "Koramangala",
                "bellandur": "Bellandur"
            }
            for alias, official in aliases.items():
                if alias in norm_query:
                    matched_loc = next((l for l in locations_list if re.sub(r'[^a-z0-9]', '', l['name'].lower()) == re.sub(r'[^a-z0-9]', '', official.lower())), None)
                    if matched_loc:
                        break

        # Default fallback locality if none mentioned
        if not matched_loc:
            matched_loc = next((l for l in locations_list if 'whitefield' in l['name'].lower()), locations_list[0])

        # 2. Extract BHK
        bhk_match = re.search(r'(\d+)\s*(?:bhk|bed|bedroom)', query)
        bhk_val = int(bhk_match.group(1)) if bhk_match else 3
        bhk_val = max(1, min(5, bhk_val))

        # 3. Extract SqFt
        sqft_match = re.search(r'(\d{3,5})\s*(?:sq\s*ft|sqft|sft)', query)
        if sqft_match:
            sqft_val = float(sqft_match.group(1))
        else:
            default_sqft_map = {1: 650, 2: 1150, 3: 1650, 4: 2400, 5: 3500}
            sqft_val = float(default_sqft_map.get(bhk_val, 1650))

        # 4. Check for Comparison Query
        is_compare = "compare" in query or " vs " in query or "versus" in query
        
        loc_name = matched_loc['name']
        price_sqft = matched_loc.get('price_sqft', 6500)
        est_val_lakhs = (sqft_val * price_sqft) / 100000
        is_premium = loc_name in premium_locations

        # Geospatial features for matched location
        schools, hospitals, dist_cbd, trend = get_geo_features(matched_loc['lat'], matched_loc['lon'])
        trend = round(trend, 1)
        inv_score = round(min(98, max(45, (trend * 8.5) + (10 if is_premium else 0))), 1)

        # Build consultative response
        if is_compare:
            reply = (
                f"### ⚖️ Micro-Market Analysis: {loc_name}\n\n"
                f"• **Benchmark Rate:** ₹{price_sqft:,.0f} / sq ft\n"
                f"• **Capital Momentum Index:** {trend}/10 ({'Prime Corridor' if trend >= 7 else 'Emerging Corridor'})\n"
                f"• **Investment Score:** {inv_score}/100\n"
                f"• **Social Infrastructure:** {schools}/10 Schools, {hospitals}/6 Hospitals within 3km\n\n"
                f"For a standard **{bhk_val}BHK ({sqft_val:,.0f} sq ft)** in {loc_name}, expect an institutional valuation of approximately **₹{est_val_lakhs:,.1f} Lakhs** (~₹{est_val_lakhs/100:.2f} Cr)."
            )
        else:
            tier_badge = "Tier-1 Prime Tech Zone" if is_premium else "Established Residential Hub"
            reply = (
                f"### 📍 Valuation Intelligence for {loc_name} ({tier_badge})\n\n"
                f"Based on Bengaluru's CatBoost pricing engine and **{loc_name}** micro-market telemetry:\n\n"
                f"• **Estimated Valuation ({bhk_val}BHK, {sqft_val:,.0f} sq ft):** **₹{est_val_lakhs:,.1f} Lakhs** (~₹{est_val_lakhs/100:.2f} Cr)\n"
                f"• **Locality Baseline:** ₹{price_sqft:,.0f} / sq ft\n"
                f"• **Investment Potential Score:** {inv_score}/100\n"
                f"• **Commute to CBD (MG Road):** ~{dist_cbd*111:.1f} km with strong arterial connectivity\n\n"
                f"💡 *Click below to automatically load this configuration into the valuation engine.*"
            )

        return jsonify({
            "success": True,
            "reply": reply,
            "action": {
                "location": loc_name,
                "lat": matched_loc['lat'],
                "lon": matched_loc['lon'],
                "price_sqft": price_sqft,
                "bhk": bhk_val,
                "bath": min(bhk_val, 4),
                "sqft": sqft_val
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------------------------------
# RUN APP
# ---------------------------------------

if __name__ == "__main__":
    app.run(debug=True)