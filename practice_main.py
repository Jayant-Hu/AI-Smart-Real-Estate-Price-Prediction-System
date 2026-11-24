from flask import Flask, render_template, request

import pickle
import pandas as pd
import numpy as np
import math

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

# ---------------------------------------
# LOAD LOCATION PRICE MAP
# ---------------------------------------

location_price_map = pd.read_csv(

    "models/location_price_map.csv"
)

location_price_map = dict(

    zip(

        location_price_map['location'],

        location_price_map['price_per_sqft']
    )
)

# ---------------------------------------
# FIND NEAREST LOCATION
# ---------------------------------------

def find_nearest_location(lat, lon):

    min_distance = float('inf')

    nearest_location = None

    for _, row in geo_data.iterrows():

        d = math.sqrt(

            (lat - row['latitude'])**2 +

            (lon - row['longitude'])**2
        )

        if d < min_distance:

            min_distance = d

            nearest_location = row['location']

    return nearest_location

# ---------------------------------------
# HOME PAGE
# ---------------------------------------

@app.route('/')
def home():

    return render_template("index.html")

# ---------------------------------------
# PREDICTION
# ---------------------------------------

@app.route('/predict', methods=['POST'])

def predict():

    try:

        # ---------------------------------------
        # FORM VALUES
        # ---------------------------------------

        lat = float(request.form['lat'])

        lon = float(request.form['lon'])

        sqft = float(request.form['total_sqft'])

        bath = float(request.form['bath'])

        bhk = float(request.form['bhk'])

        # ---------------------------------------
        # FIND LOCATION
        # ---------------------------------------

        location = find_nearest_location(
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
        # ML FEATURES
        # ---------------------------------------

        features = [[

            sqft,

            bath,

            bhk,

            estimated_pps,

            luxury_score,

            bath_per_bhk,

            lat,

            lon,

            area_avg_price
        ]]

        # ---------------------------------------
        # ML PREDICTION
        # ---------------------------------------

        ml_prediction = model.predict(features)[0]

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

            (base_price * 0.7)

            +

            (ml_prediction * 0.3)
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

        # ---------------------------------------
        # FINAL ROUND
        # ---------------------------------------

        final_price = round(final_price, 2)

        # ---------------------------------------
        # RETURN RESULT
        # ---------------------------------------

        return render_template(

            "result.html",

            prediction=final_price,

            location=location,

            sqft=sqft,

            bhk=bhk,

            bath=bath,

            price_sqft=round(
                area_avg_price,
                2
            ),

            lat=lat,

            lon=lon
        )

    except Exception as e:

        return f"ERROR: {e}"

# ---------------------------------------
# RUN APP
# ---------------------------------------

if __name__ == "__main__":

    app.run(debug=True)